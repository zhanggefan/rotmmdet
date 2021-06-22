from mmdet.models import SingleStageDetector, DETECTORS
from .showR_mixin import ShowRMixin


@DETECTORS.register_module()
class SingleStageDetectorR(ShowRMixin, SingleStageDetector):
    def __init__(self, *args, **kwargs):
        super(SingleStageDetectorR, self).__init__(*args, **kwargs)
