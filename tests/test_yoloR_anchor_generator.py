from mmdetrot.core.anchor import YOLOGaussianRAnchorGenerator


def test_yoloR_anchor_generator():
    ag = YOLOGaussianRAnchorGenerator(
        base_sizes=[
            [(10, 20), (20, 10), (14.1, 14.1)],  # P3/8
            [(20, 40), (40, 20), (28.3, 28.3)],  # P4/16
            [(40, 80), (80, 40), (56.6, 56.6)]
        ],  # P5/32
        strides=[8, 16, 32])


if __name__ == '__main__':
    test_yoloR_anchor_generator()
