# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, Optional

from mmengine.device import get_device
from mmengine.hooks.ema_hook import EMAHook
from mmengine.registry import HOOKS
from mmengine.runner import Runner

from mmdet.utils.benchmark import print_log


@HOOKS.register_module()
class EMADynamicMomentumHook(EMAHook):

    def __init__(self,
                 *args,
                 restart_epoch: int,
                 restart_momentum: Optional[float] = None,
                 interval: float = 0.0001,
                 metric: str = 'auto',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.restart_epoch = restart_epoch
        self.restart_momentum = restart_momentum
        self.interval = interval
        self.metric = metric
        self.best_epoch: Optional[int] = None
        self.best_metric: Optional[float] = None
        self.prev_metric: Optional[float] = None
        self.cur_momentum: Optional[float] = None

    def _resume_best_checkpoint(self, runner: Runner):
        # resume model and optimizer
        assert self.best_epoch is not None
        filename = os.path.join(runner.work_dir,
                                f'epoch_{self.best_epoch}.pth')
        checkpoint = runner.load_checkpoint(
            filename, map_location=get_device())
        runner.optim_wrapper.load_state_dict(checkpoint['optimizer'])

    # TODO some bug to fix

    # def after_val_epoch(self,
    #                     runner: Runner,
    #                     metrics: Optional[Dict[str, float]] = None) -> None:
    #     super().after_val_epoch(runner, metrics)

    #     if self.metric == 'auto':
    #         metric = metrics[list(metrics.keys())[0]]
    #     else:
    #         metric = metrics[self.metric]

    #     if self.best_metric is None or self.best_epoch is None:
    #         self.best_metric = metric
    #         self.best_epoch = runner.epoch

    #     # stage 1
    #     if runner.epoch <= self.restart_epoch:
    #         if metric >= self.best_metric:
    #             self.best_metric = metric
    #             self.best_epoch = runner.epoch
    #         elif runner.epoch == self.restart_epoch:
    #             self._resume_best_checkpoint(runner)
    #             if self.restart_momentum is not None:
    #                 self.ema_model.momentum = self.restart_momentum
    #             print_log(
    #                 f'Resume training from the best model: '
    #                 f'epoch_{self.best_epoch}.pth',
    #                 logger=runner.logger)

    #     # stage 2
    #     else:
    #         if self.prev_metric is None:
    #             self.prev_metric = metric

    #         if metric < max(self.prev_metric, self.best_metric):
    #             self.cur_momentum = self.ema_model.momentum
    #             self._resume_best_checkpoint(runner)
    #             self.ema_model.momentum = self.cur_momentum + self.interval
    #             print_log(
    #                 f'Change EMA momentum to {self.ema_model.momentum}, '
    #                 'and resume training from the best model: '
    #                 f'epoch_{self.best_epoch}.pth',
    #                 logger=runner.logger)

    #             self.prev_metric = self.best_metric
    #         else:
    #             self.prev_metric = metric

    # def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
    #     checkpoint['ema_dynamic_momentum_dict'] = {
    #         'restart_epoch': self.restart_epoch,
    #         'interval': self.interval,
    #         'metric': self.metric,
    #         'cur_momentum': self.cur_momentum or self.ema_model.momentum,
    #         'best_epoch': self.best_epoch,
    #         'best_metric': self.best_metric,
    #         'prev_metric': self.prev_metric
    #     }
    #     super().before_save_checkpoint(runner, checkpoint)

    # def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
    #     if 'ema_dynamic_momentum_dict' in checkpoint and runner._resume:
    #         dynamic_momentum_dict = checkpoint['ema_dynamic_momentum_dict']
    #         self.restart_epoch = dynamic_momentum_dict['restart_epoch']
    #         self.interval = dynamic_momentum_dict['interval']
    #         self.metric = dynamic_momentum_dict['metric']
    #         self.cur_momentum = dynamic_momentum_dict['cur_momentum']
    #         self.best_epoch = dynamic_momentum_dict['best_epoch']
    #         self.best_metric = dynamic_momentum_dict['best_metric']
    #         self.prev_metric = dynamic_momentum_dict['prev_metric']
    #         print_log(
    #             f'Resuming EMA dynamic momentum with: '
    #             f'restart_epoch={self.restart_epoch}, '
    #             f'interval={self.interval}, metric={self.metric}, '
    #             f'cur_momentum={self.cur_momentum}, '
    #             f'best_epoch={self.best_epoch}, '
    #             f'best_metric={self.best_metric}, '
    #             f'prev_metric={self.prev_metric}',
    #             logger=runner.logger)

    #         self.ema_model.momentum = self.cur_momentum

    #     super().after_load_checkpoint(runner, checkpoint)
