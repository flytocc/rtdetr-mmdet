# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from .deim import DEIMDFINE


@MODELS.register_module()
class RTDETRV4(DEIMDFINE):
    r"""Implementation of `RT-DETRv4: Painlessly Furthering Real-Time Object
    Detection with Vision Foundation Models <https://arxiv.org/pdf/2510.25257>`_

    Code is modified from the `official github repo
    <https://github.com/RT-DETRs/RT-DETRv4>`_.
    """
