import torch

from .base_bbox_coder import BaseBBoxCoder
from ..builder import BBOX_CODERS
import numpy as np


def norm_angle(rbboxes, ag_range=(-np.pi/2, np.pi)):
    angle = (rbboxes[:, 4:5] + ag_range[0]
             ) % (-ag_range[1]) - ag_range[0]

    rbboxes_norm = torch.where(
        angle.gt(0).expand(-1, 5),
        torch.cat([rbboxes[:, :2],
                   rbboxes[:, 3:4],
                   rbboxes[:, 2:3],
                   angle - (np.pi/2)],
                  dim=1),
        torch.cat([rbboxes[:, :4],
                   angle],
                  dim=1)
    )
    return rbboxes_norm


def bbox2delta_rotated(proposals, gt, means=(0., 0., 0., 0., 0.),
                       stds=(1., 1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 5)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 5), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()

    # fist, gt norm to cv2 formula?
    gt = norm_angle(gt)

    px, py, pw, ph, pa = proposals.split(1, dim=1)
    gx, gy, gw, gh, ga = gt.split(1, dim=1)

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    # da = (ga - pa)
    da = (ga - pa) / (np.pi / 2)

    deltas = torch.cat([dx, dy, dw, dh, da], dim=1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox_rotated(rois, deltas, means=(0., 0., 0., 0., 0.),
                       stds=(1., 1., 1., 1., 1.), max_shape=None,
                       wh_ratio_clip=16 / 1000, clip_border=True):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 5)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 5 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.

    Returns:
        Tensor: Boxes with shape (N, 5), where columns represent

    References:
        .. [1] https://arxiv.org/abs/1311.2524
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    da = denorm_deltas[:, 4::5]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = rois[:, 0].unsqueeze(1).expand_as(dx)
    py = rois[:, 1].unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = rois[:, 2].unsqueeze(1).expand_as(dw)
    ph = rois[:, 3].unsqueeze(1).expand_as(dh)
    # Compute rotated angle of each roi
    pa = rois[:, 4].unsqueeze(1).expand_as(da)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy

    # Compute angle
    # ga = pa + da
    ga = pa + da * (np.pi / 2)

    if max_shape is not None:
        gx = gx.clamp(min=0, max=max_shape[1])
        gy = gy.clamp(min=0, max=max_shape[0])
    rbboxes = torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)
    rbboxes = norm_angle(rbboxes)
    return rbboxes


@BBOX_CODERS.register_module
class DeltaXYWHACV2BBoxCoder(BaseBBoxCoder):
    """Delta XYWHA BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x,y,w,h,a) into delta (dx, dy, dw, dh,da) and
    decodes delta (dx, dy, dw, dh,da) back to original bbox (x, y, w, h, a).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 clip_border=True):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 5
        encoded_bboxes = bbox2delta_rotated(bboxes, gt_bboxes,
                                            self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox_rotated(bboxes, pred_bboxes, self.means, self.stds,
                                            max_shape, wh_ratio_clip, self.clip_border)

        return decoded_bboxes
