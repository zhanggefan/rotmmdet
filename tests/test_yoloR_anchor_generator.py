from mmdetrot.core.anchor import YOLOGaussianRAnchorGenerator
import torch


def test_yoloR_anchor_generator():
    ag = YOLOGaussianRAnchorGenerator(
        base_sizes=[
            [(10, 20), (20, 10), (14.1, 14.1)],  # P3/8
            [(20, 40), (40, 20), (28.3, 28.3)],  # P4/16
            [(40, 80), (80, 40), (56.6, 56.6)]
        ],  # P5/32
        strides=[8, 16, 32])
    featmap_sizes = [(60, 120), (30, 60), (15, 30)]
    gt_bboxes_list = [torch.tensor([
        [0., 0., 30., 30., 0.],
        [960., 480., 30., 30., 0.],
        [960., 0., 30., 30., 0.],
        [0., 480., 30., 30., 0.],
        [480., 480., 30., 30., 0.],
        [960., 240., 30., 30., 0.]])]
    ri = ag.responsible_indices(featmap_sizes, gt_bboxes_list)
    for i in range(3):
        assert (ri[i][0] == 0).all()
        x = (ri[i][1] // 3) % featmap_sizes[i][1]
        y = (ri[i][1] // 3) // featmap_sizes[i][1]
        assert x.min() >= 0
        assert x.max() < featmap_sizes[i][1]
        assert y.min() >= 0
        assert y.max() < featmap_sizes[i][0]


if __name__ == '__main__':
    test_yoloR_anchor_generator()
