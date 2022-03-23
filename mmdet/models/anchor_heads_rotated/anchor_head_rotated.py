from __future__ import division

import torch
import torch.nn as nn

from mmdet.core import (AnchorGeneratorRotated, delta2bbox_rotated, multiclass_nms_rotated,
                        anchor_target, delta2bbox, force_fp32, multi_apply, multiclass_nms,
                        build_bbox_coder, images_to_levels)
from ..anchor_heads import AnchorHead
from ..registry import HEADS
import numpy as np

@HEADS.register_module
class AnchorHeadRotated(AnchorHead):

    def __init__(self, *args, anchor_angles=[0., ], **kargs):
        super(AnchorHeadRotated, self).__init__(*args, **kargs)

        self.anchor_angles = anchor_angles

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGeneratorRotated(
                    anchor_base, self.anchor_scales, self.anchor_ratios, angles=anchor_angles))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales) * len(self.anchor_angles)

        self._init_layers()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.in_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 5, 1)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, anchor,
                    num_total_samples, cfg):
        labels = labels.reshape(-1)

        pos_inds = labels.gt(0)
        if pos_inds.any():
            # regression loss
            bbox_targets = bbox_targets.reshape(-1, 5)
            bbox_weights = bbox_weights.reshape(-1, 5)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
            if not cfg.get('target_encode', True) and not cfg.get('reg_decoded_bbox', False):
                bbox_coder_cfg = cfg.get('bbox_coder', dict(type='DeltaXYWHBBoxCoder'))
                bbox_coder = build_bbox_coder(bbox_coder_cfg)

                pos_target = bbox_targets[pos_inds, :]
                pos_pred = bbox_pred[pos_inds, :]
                pos_weights = bbox_weights[pos_inds, :]
                pos_anchor = anchor.reshape(-1, 5)[pos_inds, :]
                loss_bbox = self.loss_bbox(
                    pos_pred,
                    pos_target,
                    pos_anchor,
                    avg_factor=num_total_samples,
                    bbox_coder=bbox_coder,
                    weights=pos_weights)
            else:
                if cfg.get('reg_decoded_bbox', False):
                    bbox_coder_cfg = cfg.get('bbox_coder', dict(type='DeltaXYWHBBoxCoder'))
                    bbox_coder = build_bbox_coder(bbox_coder_cfg)
                    bbox_pred = bbox_coder.decode(anchor.reshape(-1, 5), bbox_pred)
                loss_bbox = self.loss_bbox(
                    bbox_pred,
                    bbox_targets,
                    bbox_weights,
                    avg_factor=num_total_samples)

        else:
            loss_bbox = bbox_pred.sum() * 0.

        # classification loss
        if not cfg.assigner.get('ext_data', False):
            label_weights = label_weights.reshape(-1)
        else:
            label_weights = label_weights.reshape(-1, self.cls_out_channels)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)

        # compare with different coder
        bbox_coder = cfg.get('bbox_coder', None)
        if bbox_coder is not None:
            bbox_coder = build_bbox_coder(bbox_coder)

        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            if bbox_coder is None:
                bboxes = delta2bbox_rotated(anchors, bbox_pred, self.target_means,
                                            self.target_stds, img_shape)
            else:
                bboxes = bbox_coder.decode(anchors, bbox_pred, img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes[..., :4] /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms_rotated(mlvl_bboxes, mlvl_scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
        return det_bboxes, det_labels

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device

        ext_data = [[]] * cls_scores[0].shape[0]
        if cfg.assigner.get('ext_data', False):
            ext_data = [cls_scores, bbox_preds]
            for e in range(len(ext_data)):
                b, c, _, _ = ext_data[e][0].shape
                ext_data[e] = torch.cat([ext_data[e][i].detach(
                    ).permute(0, 2, 3, 1).reshape(b, -1, c//self.num_anchors)
                    for i in range(len(ext_data[e]))],
                                        dim=1).unbind(dim=0)
            ext_data = list(zip(*ext_data))

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            ext_data=ext_data)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # concat all level anchors and flags to a single tensor
        all_anchor_list = images_to_levels(anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            all_anchor_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)