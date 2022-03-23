import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from ..builder import build_bbox_coder
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from ..transforms_rotated import pts_in_rect


@BBOX_ASSIGNERS.register_module
class DALAssigner(BaseAssigner):
    """
    https://github.com/ming71/DAL/blob/master/models/losses.py
    """
    def __init__(self, das=False, md_thres=0.7,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 alpha=0.3, var=5., match_low_quality=True,
                 ext_data=True):
        self.das = das
        self.alpha = alpha
        self.var = var
        self.md_thres = md_thres
        self.match_low_quality = match_low_quality
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ext_data = ext_data

    def get_train_param(self, iter_now, iter_max):
        process = iter_now / iter_max
        if process < 0.1:
            bf_weight = 1.0
        elif process > 0.3:
            bf_weight = self.alpha
        else:
            bf_weight = 5*(self.alpha-1)*process+1.5-0.5*self.alpha
        return bf_weight

    def assign(self,
               bf_bbox,
               pred_out,
               bbox_coder,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               img_meta=None):
        bf_bbox = bf_bbox[:, :5]
        # [anchor_n, cls_n]
        cls_pred = pred_out[0]
        # [anchor_n, 5]
        delta_pred = pred_out[1][:, :5]
        num_gt, num_bboxes = gt_bboxes.size(0), bf_bbox.size(0)

        # compute iou between all bf_bbox and gt
        bf_overlaps = self.iou_calculator(bf_bbox, gt_bboxes)

        af_bbox = bbox_coder.decode(bf_bbox, delta_pred)
        # compute iou between all af_bbox and gt
        af_overlaps = self.iou_calculator(af_bbox, gt_bboxes)

        bf_weight = self.get_train_param(img_meta['iter_now'],
                                         img_meta['iter_max'])
        alpha, beta, var = bf_weight, 1 - bf_weight, self.var

        if var != -1:
            if var == 0:
                md = abs((alpha * bf_overlaps + beta * af_overlaps))
            else:
                # compute iou coeff
                md = (alpha * bf_overlaps + beta * af_overlaps
                      - (af_overlaps - bf_overlaps).abs() ** var).abs()
        else:
            self.das = False
            md = bf_overlaps
        del bf_overlaps, af_overlaps

        # [a_n, ], anchor match gt
        anchor_max, anchor_argmax = md.max(dim=1)
        # [gt_n, ], gt match anchor
        gt_max, gt_argmax = md.max(dim=0)

        # 1, assign -1 by default, -1 indicate ignore,
        # (self.md_thres - 0.1 < ignore < self.md_thres)
        assigned_gt_inds = anchor_max.new_full((num_bboxes, ),
                                               -1,
                                               dtype=torch.long)
        # 2, anchor_max < self.md_thres - 0.1 is bg
        # bg < self.md_thres - 0.1
        assigned_gt_inds[anchor_max.lt(self.md_thres - 0.1)] = 0

        # 3. iou_coeff > thres is pos, [a_n, ]
        positive_indices = anchor_max.ge(self.md_thres)
        # if some gt can not has a matched anchor that iou > thres,
        # we allow the low quality match, so every gt match a anchor
        if self.match_low_quality:  #  and (gt_max < self.md_thres).any():
            # set unique low quality match location 1, [a_n, ]
            positive_indices[gt_argmax[gt_max < self.md_thres]] = 1
        assigned_gt_inds[positive_indices] = anchor_argmax[positive_indices] + 1

        assigned_labels = assigned_gt_inds.new_zeros((num_bboxes,))
        pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[
                assigned_gt_inds[pos_inds] - 1]

        # matching-weight
        if self.das:
            # select all pos, [select_n, gt_n]
            pos_coeff = md[positive_indices]
            pos_mask = pos_coeff.ge(self.md_thres)
            # [gt_n, ]
            max_pos_coeff, armmax_pos_coeff = pos_coeff.max(0)
            nt = md.shape[1]
            # match a
            for gt_idx in range(nt):
                pos_mask[armmax_pos_coeff[gt_idx], gt_idx] = 1
            comp = torch.where(pos_mask,
                               (1 - max_pos_coeff).repeat(len(pos_coeff), 1),
                               pos_coeff)
            matching_weight = comp + pos_coeff
            # #########################################cls weight ######################################
            # cls_targets, cls_pred = [a_n, cls_n]
            # default ignore
            cls_targets = cls_pred.new_full(cls_pred.shape, -1, dtype=torch.long)
            # bg
            cls_targets[anchor_max.lt(self.md_thres - 0.1), :] = 0

            # assigned_annotations = gt_bboxes[anchor_argmax, :]
            cls_targets[positive_indices, :] = 0

            # 0 is bg, and focal use sigmoid cls, so delete 0
            labels_anchor = gt_labels[anchor_argmax] - 1

            cls_targets[positive_indices,
                        labels_anchor[positive_indices].long()] = 1
            soft_weight = torch.zeros_like(cls_pred)
            soft_weight = torch.where(torch.eq(cls_targets, 0),
                                      torch.ones_like(cls_pred),  # bg weight = 1
                                      soft_weight)

            # fg weight = max(matching_weight) + 1
            soft_weight[positive_indices,
                        labels_anchor[positive_indices].long()] = (
                        matching_weight.max(1)[0] + 1)

            # ignore = 0
            soft_weight = torch.where(torch.ne(cls_targets, -1.0),
                                      soft_weight,
                                      cls_pred.new_tensor(0))
            self.weight = dict(cls_weight=soft_weight,
                               bbox_weight=matching_weight.max(1)[0].unsqueeze(1).repeat(1, 5))
        return AssignResult(
            num_gt, assigned_gt_inds, anchor_max, labels=assigned_labels)
