# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import CheckpointHook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class CheckpointAfterValHook(CheckpointHook):

    def after_train_epoch(self, runner) -> None:
        return

    def after_val_epoch(self, runner, metrics):
        assert self.by_epoch, \
            'Only support `by_epoch=True` in CheckpointAfterValHook.'

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs which start at ``self.save_begin``
        # 2. reach the last epoch of training
        if self.every_n_epochs(runner, self.interval, self.save_begin +
                               1) or (self.save_last
                                      and self.is_last_train_epoch(runner)):
            runner.logger.info(f'Saving checkpoint at {runner.epoch} epochs')
            step = runner.epoch
            meta = dict(epoch=step, iter=runner.iter)
            self._save_checkpoint_with_step(runner, step, meta=meta)

        super().after_val_epoch(runner, metrics)
