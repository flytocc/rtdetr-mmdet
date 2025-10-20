# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import CheckpointHook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class CheckpointAfterValHook(CheckpointHook):

    def after_train_epoch(self, runner) -> None:
        return

    def after_val_epoch(self, runner, metrics):
        runner.epoch -= 1
        super().after_train_epoch(runner)
        runner.epoch += 1
        super().after_val_epoch(runner, metrics)
