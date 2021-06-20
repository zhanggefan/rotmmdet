from shapely.geometry import Polygon
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmdetrot.core.bbox.iou_calculators.iou2dR_calculator import box_iou_rotated


def rbbox_4pts(xywhr):
    xywhr = xywhr.cpu()
    xy, w, h, r = xywhr[..., :2], xywhr[..., 2:3], xywhr[..., 3:4], xywhr[
        ..., 4]
    sinr, cosr = torch.sin(r), torch.cos(r)
    dir1 = torch.stack((cosr, sinr), dim=-1)
    dir2 = torch.stack((-sinr, cosr), dim=-1)
    w /= 2
    h /= 2
    pts1 = dir1 * w + dir2 * h + xy
    pts2 = -dir1 * w + dir2 * h + xy
    pts3 = -dir1 * w - dir2 * h + xy
    pts4 = dir1 * w - dir2 * h + xy
    return torch.stack((pts1, pts2, pts3, pts4), dim=-2)


def plot_rbbox(xywhr):
    xywhr = xywhr.cpu()
    xy, w, h, r = xywhr[..., :2], xywhr[..., 2:3], xywhr[..., 3:4], xywhr[
        ..., 4]
    sinr, cosr = torch.sin(r), torch.cos(r)
    dir1 = torch.stack((cosr, sinr), dim=-1)
    dir2 = torch.stack((-sinr, cosr), dim=-1)
    w /= 2
    h /= 2
    pts1 = dir1 * w + dir2 * h + xy
    pts2 = -dir1 * w + dir2 * h + xy
    pts3 = -dir1 * w - dir2 * h + xy
    pts4 = dir1 * w - dir2 * h + xy
    pts_loop = torch.stack((pts1, pts2, pts3, pts4, pts1), dim=0)
    pts_x = pts_loop[..., 0]
    pts_y = pts_loop[..., 1]

    plt.plot(pts_x, pts_y)
    plt.axis('equal')
    plt.show()


def test_rbbox_iou():
    num_box = 10
    xy = torch.randn((num_box, 2)).cuda() * 5
    wh = torch.rand((num_box, 2)).cuda() * 5 + 1
    r = torch.randn((num_box, 1)).cuda() * np.pi
    rbbox = torch.cat((xy, wh, r), dim=-1)

    # plot_rbbox(rbbox)

    iou = box_iou_rotated(rbbox, rbbox, mode='iou')
    iof = box_iou_rotated(rbbox, rbbox, mode='iof')

    iou_tgt = torch.empty((num_box, num_box), dtype=torch.float32)
    iof_tgt = torch.empty((num_box, num_box), dtype=torch.float32)
    p4s = rbbox_4pts(rbbox)
    for i in range(num_box):
        for j in range(num_box):
            p4i, p4j = p4s[i], p4s[j]
            pi = Polygon(p4i.cpu())
            pj = Polygon(p4j.cpu())
            inter = pi.intersection(pj).area
            union = pi.area + pj.area - inter
            iou_tgt[i, j] = inter / union
            iof_tgt[i, j] = inter / pi.area

    assert torch.allclose(iou.cpu(), iou_tgt, rtol=1e-03, atol=1e-05)
    assert torch.allclose(iof.cpu(), iof_tgt, rtol=1e-03, atol=1e-05)


if __name__ == '__main__':
    test_rbbox_iou()
