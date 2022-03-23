import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..registry import LOSSES
from mmdet.core import rbbox2points


def smooth_l1_loss(pred, target, beta=1.0, cond=-1.):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                        diff - 0.5 * beta)
    return loss


def bbox2points(bboxes):
    """
    Args:
        bboxes (Tensor): shape (n, 4), xyxy encoded

    Returns:
        bboxes (Tensor): shape (n, 8), x1y1x2y2x3y3x4y4
    """
    x1, y1, x2, y2 = bboxes.split(1, dim=1)
    return torch.cat([x1, y1, x2, y1, x2, y2, x1, y2], dim=1)


def polygon_area(x, y):
    """
    Using the shoelace formula
    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    Args:
        x: [n, m]
        y: [n, m]

    Returns:

    """
    # [n, ]
    correction = x[:, -1] * y[:, 0] - y[:, -1] * x[:, 0]
    
    main_area = torch.einsum("ij,ij->i", x[:, :-1], y[:, 1:]) - \
                torch.einsum("ij,ij->i", y[:, :-1], x[:, 1:])
    area = 0.5 * (main_area + correction).abs()
    return area


def p2p_area(bboxes_pred, bboxes_gt, rotated=False):
    """
    pull_loss
    :param pts: [n, 4?5?]
    :param bboxes_gt: [n, 4?5?]
    :param rotated: bool
    :return:
    """
    if bboxes_gt.shape[1] == 5:
        bboxes_gt = bboxes_gt if rotated else bboxes_gt[:, :4]
    elif bboxes_gt.shape[1] == 4:
        rotated = False
    else:
        raise ValueError

    # [n, 4, 2]
    if rotated:
        pts_gt = rbbox2points(bboxes_gt).view(-1, 4, 2)
        pts_pred = rbbox2points(bboxes_pred).view(-1, 4, 2)
    else:
        pts_gt = bbox2points(bboxes_gt).view(-1, 4, 2)
        pts_pred = bbox2points(bboxes_pred).view(-1, 4, 2)

    # [n, 4, 2] -> [n, 4, 1, 2]
    pts_pred = pts_pred[:, :, None, :]
    # [n, 4, 2] -> [n, 1, 4, 2]
    pts_gt = pts_gt[:, None, :, :]
    # [n, 1, 4, 2]
    pts_gt_roll = torch.roll(pts_gt, -1, 2)
    v_tt = pts_gt_roll - pts_gt

    # [n, 4, 4, 2]
    v_tp = pts_gt - pts_pred
    v_tt = v_tt.expand_as(v_tp)

    # [n, 4, 4, 2] -> [n, 4, 4, 3]
    v_tp = F.pad(v_tp, [0, 1])
    v_tt = F.pad(v_tt, [0, 1])

    # 叉积, [n, 4, 4, 3] -> [n, 4, 4, 3] -> [n, 4, 4] -> [n,]
    cross_area = torch.cross(v_tp, v_tt, dim=3).sum(-1).abs().sum([1, 2])
    return cross_area


def Semiperimeter_or_area(bboxes_pred, bboxes_gt, rotated=False, method='1'):
    if bboxes_gt.shape[1] == 5:
        bboxes_gt = bboxes_gt if rotated else bboxes_gt[:, :4]
    elif bboxes_gt.shape[1] == 4:
        rotated = False
    else:
        raise NotImplementedError

    # [n, 4, 2]
    if rotated:
        gt_area = bboxes_gt[:, 2] * bboxes_gt[:, 3]
        if method == '1':
            pred_area = bboxes_pred[:, 2] * bboxes_pred[:, 3]
        elif method == '2':
            pred_area = bboxes_pred[:, 2] + bboxes_pred[:, 3]
        elif method == '3':
            pts_pred = rbbox2points(bboxes_pred).view(-1, 4, 2)
            pred_area = polygon_area(pts_pred[:, 0::2],
                                     pts_pred[:, 1::2])
        else:
            raise NotImplementedError
    else:
        gt_area = (bboxes_gt[:, 2:] - bboxes_gt[:, :2])
        gt_area = gt_area[:, 0] * gt_area[:, 1]
        if method == '1':
            pred_area = bboxes_pred[:, 2:] - bboxes_pred[:, :2]
            pred_area = pred_area[:, 0] * pred_area[:, 1]
        elif method == '2':
            pred_area = bboxes_pred[:, 2] + bboxes_pred[:, 3]
        elif method == '3':
            pts_pred = bbox2points(bboxes_pred)
            pred_area = polygon_area(pts_pred[:, 0::2],
                                     pts_pred[:, 1::2])
        else:
            raise NotImplementedError

    return pred_area, gt_area


def p2p_loss_v2(pred_encode, target, anchor,
                  eps=1e-6, beta=1./9, rotated=False,
                  method='2', subweight=(1., 1., 1.), **kwargs):
    bbox_coder = kwargs.get('bbox_coder')
    pred_decode = bbox_coder.decode(anchor, pred_encode)

    semip, target_area = Semiperimeter_or_area(
        pred_decode, target, rotated, method=method)
    semip = semip.clamp(min=eps)
    target_area = target_area.clamp(min=eps)

    p2p_area1 = p2p_area(pred_decode, target, rotated)
    p2p_area2 = p2p_area(target, pred_decode, rotated)

    # center+wh: smoothl1
    target_encode = bbox_coder.encode(anchor, target)
    ct_loss = smooth_l1_loss(pred_encode[:, :2], target_encode[:, :2], beta).sum(1)
    
    coeff_loss = smooth_l1_loss(semip / (target[:, 2] + target[:, 3]),
                                torch.ones_like(semip),
                                beta)

    pred_area = (pred_decode[:, 2] * pred_decode[:, 3]).clamp(min=eps)
    
    dist = (p2p_area1 + p2p_area2 - 8 * (target_area + pred_area)) / (
            target_area + pred_area) / 8
    loss = sum([ct_loss * subweight[0],
                coeff_loss * subweight[1],
                dist * subweight[2]])

    if kwargs.get('use_weight', False):
        weight = kwargs.get('weights', torch.ones_like(loss))
        if weight.ndim == 2:
            weight = weight[:, 0]
        loss *= weight
    return loss.mean()


def chamferloss(x, y, reduction=None):
    if x.shape[0] == 0:
        return x.sum()
    # The Sinkhorn algorithm takes as input three variables :
    C = cost_matrix(x, y)  # Wasserstein cost function

    # compute chamfer loss
    min_x2y, _ = C.min(-1)
    d1 = min_x2y.mean(-1)
    min_y2x, _ = C.min(-2)
    d2 = min_y2x.mean(-1)
    cost = (d1 + d2) / 2.0

    if reduction == 'mean':
        cost = cost.mean()
    elif reduction == 'sum':
        cost = cost.sum()
    return cost


def cost_matrix(x, y, p=2):
    """Returns the matrix of $|x_i-y_j|^p$."""
    x_col = x.unsqueeze(-2)
    y_lin = y.unsqueeze(-3)
    C = torch.norm(x_col - y_lin, 2, -1)
    return C


@LOSSES.register_module
class P2PLoss(nn.Module):

    def __init__(self, rotated=True, method='2', eps=1e-6,
                 beta=1./9, subweight=(1., 1., 1.),
                 reduction='mean', loss_weight=1.0,
                 **kwargs):
        super(P2PLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.rotated = rotated
        self.beta = beta
        self.method = method
        self.subweight = subweight
        self.use_weight = kwargs.get('use_weight', False)
        self.version = kwargs.get('version', 'v2')

    def forward(self,
                pred,
                target,
                anchor,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.version == 'v2':
            loss = self.loss_weight * p2p_loss_v2(
                pred,
                target,
                anchor,
                beta=self.beta,
                eps=self.eps,
                rotated=self.rotated,
                method=self.method,
                subweight=self.subweight,
                reduction=reduction,
                avg_factor=avg_factor,
                use_weight=self.use_weight,
                **kwargs)
        else:
            raise NotImplementedError
        return loss

