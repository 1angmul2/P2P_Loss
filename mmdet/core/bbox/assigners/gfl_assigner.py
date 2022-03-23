import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from ..transforms_rotated import (pts_in_rect, rbbox2points,
                                  points2minbbox)


@BBOX_ASSIGNERS.register_module
class GFLAssigner(BaseAssigner):

    def __init__(self, topk, ext_data=True, iou_calculator=dict(type='BboxOverlaps2D')):
        self.topk = topk
        self.ext_data = ext_data
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self,
               bboxes,
               num_level_bboxes,
               ext_data,
               bbox_coder,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               img_meta=None):
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)


@BBOX_ASSIGNERS.register_module
class RGFLAssigner(GFLAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self, use_hbbox=False, **kwargs):
        self.use_hbbox = use_hbbox
        super(RGFLAssigner, self).__init__(**kwargs)
        if use_hbbox:
            self.iou_calculator = build_iou_calculator(
                dict(type='BboxOverlaps2D'))

    def assign(self,
               bboxes,
               num_level_bboxes,
               ext_data,
               bbox_coder,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               img_meta=None):
        # [anchor_n, cls_n]
        cls_pred = ext_data[0]
        # [anchor_n, 5]
        delta_pred = ext_data[1][:, :5]

        INF = 100000000
        if self.use_hbbox:
            raise NotImplementedError

        bboxes = bboxes[:, :5]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)

            # init ones overlaps
            new_overlaps = torch.ones_like(max_overlaps)

            # cls weight and bbox weight
            self.weight = dict(
                cls_weight=new_overlaps[
                    ..., None].expand(-1, cls_pred.shape[1]),
                bbox_weight=new_overlaps.new_ones((0, 5))
            )
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        gt_points = gt_bboxes[:, :2]
        bboxes_points = bboxes[:, :2]

        distances = (bboxes_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)
        del distances

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)
        ep_pts = bboxes_points.view(
            1, -1, 2).expand(num_gt, -1, -1).reshape(-1, 2)[candidate_idxs]

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        is_in_gts = pts_in_rect(
            ep_pts.view(-1, num_gt, 2), gt_bboxes)
        del ep_pts
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            # use squeeze() maybe cause miss dim 1 in delta_pred[pos_inds, :]
            # so use [:, 0]
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False)[:, 0]
            if pos_inds.numel() > 0:
                bbox_match_gt = assigned_gt_inds[pos_inds] - 1
                assigned_labels[pos_inds] = gt_labels[bbox_match_gt]

                bbox_pred = bbox_coder.decode(bboxes[pos_inds, :],
                                              delta_pred[pos_inds, :])

                # compute iou between pos af_bbox and gt, [pos_num, gt_n]
                af_overlaps = self.iou_calculator(bbox_pred, gt_bboxes)
                # select match overlaps
                af_overlaps = af_overlaps[range(pos_inds.numel()), bbox_match_gt]

                # init ones overlaps
                new_overlaps = torch.ones_like(max_overlaps)
                new_overlaps[pos_inds] = af_overlaps

                # cls weight and bbox weight
                self.weight = dict(
                    cls_weight=new_overlaps[
                        ..., None].expand(-1, cls_pred.shape[1]),
                    bbox_weight=af_overlaps[..., None].expand(-1, 5)
                )
            else:
                # init ones overlaps
                new_overlaps = torch.ones_like(max_overlaps)
                self.weight = dict(
                    cls_weight=new_overlaps[
                        ..., None].expand(-1, cls_pred.shape[1]),
                    bbox_weight=max_overlaps.new_zeros((0, 5))
                )
        else:
            assigned_labels = None

        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
