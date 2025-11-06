# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, Optional

from mmengine.hooks.ema_hook import EMAHook
from mmengine.registry import HOOKS
from mmengine.runner import Runner, load_checkpoint
from mmdet.utils.benchmark import print_log


@HOOKS.register_module()
class EMADynamicMomentumHook(EMAHook):
    """EMADynamicMomentumHook. This hook implements a two-stage training
    strategy with dynamic EMA momentum adjustment based on validation
    performance.

    Args:
        restart_epoch (int): The epoch to restart training from the best model
            found in stage 1.
        restart_momentum (float, optional): The momentum to be set at the
            beginning of stage 2. If None, keep the momentum unchanged.
            Defaults to None.
        interval (float): The interval to increase EMA momentum when no
            improvement is observed in stage 2. Defaults to 0.0001.
        metric (str): The metric to monitor. Defaults to 'auto'.

    Note:
        This hook requires `CheckpointAfterValHook` to be also registered to
        the runner, and `CheckpointAfterValHook.after_val_epoch` must be called
        after `EMADynamicMomentumHook.after_val_epoch`.

        1. In stage 1, `after_val_epoch` records the best model based on the
        specified metric, `before_train_epoch` does nothing.
        2. At the beginning of stage 2, `before_train_epoch` resumes training
        from the best model found in stage 1 and resets the EMA momentum to
        `restart_momentum` if provided, `after_val_epoch` initializes the best
        metric for stage 2.
        3. During stage 2, if no improvement is observed in the specified
        metric compared to the best metric found in stage 2, the hook resumes
        training from the best model found in stage 1 and increases the EMA
        momentum by `interval`.
    """

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
        self.cur_metric: Optional[float] = None
        self.best_epoch_stage1: Optional[int] = None
        self.best_metric_stage1: Optional[float] = None
        self.best_metric_stage2: Optional[float] = None
        self.best_epoch: Optional[float] = None
        self.best_metric: Optional[float] = None

    def _resume_best_checkpoint(self, runner: Runner):
        assert self.best_epoch_stage1 is not None
        filename = os.path.join(runner.work_dir,
                                f'epoch_{self.best_epoch_stage1}.pth')
        # model
        checkpoint = load_checkpoint(
            runner.model, filename, map_location='cpu', strict=True)
        # optim_wrapper
        runner.optim_wrapper.load_state_dict(checkpoint['optimizer'])
        # ema
        self._load_ema_dynamic_momentum_dict(runner, checkpoint)
        assert 'ema_state_dict' in checkpoint
        # The original model parameters are actually saved in ema
        # field swap the weights back to resume ema state.
        self._swap_ema_state_dict(checkpoint)
        self.ema_model.load_state_dict(
            checkpoint['ema_state_dict'], strict=self.strict_load)

    def before_train_epoch(self, runner: Runner) -> None:
        """
        Note:
            runner.epoch starts from 0.
        """
        super().before_train_epoch(runner)

        # stage 1
        if runner.epoch < self.restart_epoch:
            return

        old_momentum = self.ema_model.momentum

        # at the beginning of stage 2
        # resume best checkpoint and reset momentum
        if runner.epoch == self.restart_epoch:
            if runner.epoch != self.best_epoch_stage1:
                print_log(
                    f'Refresh training from the best model: '
                    f'epoch_{self.best_epoch_stage1}.pth',
                    logger=runner.logger)
                self._resume_best_checkpoint(runner)
            if self.restart_momentum is not None and \
                    self.ema_model.momentum != self.restart_momentum:
                print_log(
                    f'Change EMA momentum from {old_momentum} '
                    f'to {self.restart_momentum}',
                    logger=runner.logger)
                self.ema_model.momentum = self.restart_momentum

        # after the beginning of stage 2
        else:
            if self.cur_metric < self.best_metric_stage2:
                print_log(
                    f'Refresh training from the best model: '
                    f'epoch_{self.best_epoch_stage1}.pth',
                    logger=runner.logger)
                self._resume_best_checkpoint(runner)

                print_log(
                    f'Change EMA momentum from {old_momentum} '
                    f'to {old_momentum + self.interval}',
                    logger=runner.logger)
                self.ema_model.momentum = old_momentum + self.interval

    def after_val_epoch(self,
                        runner: Runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Note:
            1. runner.epoch starts from 1.
            2. CheckpointAfterValHook.after_val_epoch must be called after here
        """

        super().after_val_epoch(runner, metrics)

        metric = list(metrics.keys())[0] \
            if self.metric == 'auto' else self.metric
        self.cur_metric = metrics[metric]

        if self.best_metric is None or self.cur_metric >= self.best_metric:
            self.best_metric = self.cur_metric
            self.best_epoch = runner.epoch

        # stage 1
        if runner.epoch <= self.restart_epoch:
            if self.best_metric_stage1 is None or \
                    self.cur_metric >= self.best_metric_stage1:
                self.best_metric_stage1 = self.cur_metric
                self.best_epoch_stage1 = runner.epoch

        # stage 2
        else:
            if self.best_metric_stage2 is None or \
                    self.cur_metric >= self.best_metric_stage2:
                self.best_metric_stage2 = self.cur_metric

    def before_save_checkpoint(self, runner: Runner, checkpoint: Dict) -> None:
        assert self.cur_metric is not None, (
            'cur_metric is None, please make sure to save checkpoint after '
            'validation.')
        assert self.best_epoch is not None, (
            'best_epoch is None, please make sure to save checkpoint after '
            'validation.')
        assert self.best_metric is not None, (
            'best_metric is None, please make sure to save checkpoint after '
            'validation.')
        assert self.best_epoch_stage1 is not None, (
            'best_epoch_stage1 is None, please make sure to save checkpoint '
            'after validation.')
        assert self.best_metric_stage1 is not None, (
            'best_metric_stage1 is None, please make sure to save checkpoint '
            'after validation.')

        if runner.epoch <= self.restart_epoch:
            assert self.best_metric_stage2 is None, (
                'best_metric_stage2 is not None in stage 1, please make sure '
                'to save checkpoint after validation.')
        else:
            assert self.best_metric_stage2 is not None, (
                'best_metric_stage2 is None in stage 2, please make sure to '
                'save checkpoint after validation.')

        checkpoint['ema_dynamic_momentum_dict'] = {
            'restart_epoch': self.restart_epoch,
            'interval': self.interval,
            'metric': self.metric,
            'cur_metric': self.cur_metric,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'best_epoch_stage1': self.best_epoch_stage1,
            'best_metric_stage1': self.best_metric_stage1,
            'best_metric_stage2': self.best_metric_stage2,
            'momentum': self.ema_model.momentum,
        }
        super().before_save_checkpoint(runner, checkpoint)

    def _load_ema_dynamic_momentum_dict(self, runner: Runner, checkpoint: Dict) -> None:
        if 'ema_dynamic_momentum_dict' in checkpoint:
            dynamic_momentum_dict = checkpoint['ema_dynamic_momentum_dict']
            assert self.restart_epoch == dynamic_momentum_dict['restart_epoch'], (
                f"Inconsistent restart_epoch: current={self.restart_epoch}, "
                f"checkpoint={dynamic_momentum_dict['restart_epoch']}")
            assert self.metric == dynamic_momentum_dict['metric'], (
                f"Inconsistent metric: current={self.metric}, "
                f"checkpoint={dynamic_momentum_dict['metric']}")

            if self.interval != dynamic_momentum_dict['interval']:
                print_log(
                    f'Warning: interval changed: current={self.interval}, '
                    f"checkpoint={dynamic_momentum_dict['interval']}",
                    logger=runner.logger)
            self.interval = dynamic_momentum_dict['interval']

            assert dynamic_momentum_dict['cur_metric'] is not None, (
                'cur_metric in the checkpoint is None, please make sure the '
                'checkpoint is saved after validation.')
            self.cur_metric = dynamic_momentum_dict['cur_metric']

            assert dynamic_momentum_dict[
                'best_epoch'] is not None, (
                'best_epoch in the checkpoint is None, please make sure the '
                'checkpoint is saved after validation.')
            self.best_metric = dynamic_momentum_dict['best_metric']

            assert self.best_metric is not None, (
                'best_metric in the checkpoint is None, please make sure the '
                'checkpoint is saved after validation.')
            self.best_epoch = dynamic_momentum_dict['best_epoch']

            assert dynamic_momentum_dict[
                'best_epoch_stage1'] is not None, (
                'best_epoch_stage1 in the checkpoint is None, please make sure '
                'the checkpoint is saved after validation.')
            self.best_epoch_stage1 = dynamic_momentum_dict['best_epoch_stage1']

            assert dynamic_momentum_dict[
                'best_metric_stage1'] is not None, (
                'best_metric_stage1 in the checkpoint is None, please make sure '
                'the checkpoint is saved after validation.')
            self.best_metric_stage1 = dynamic_momentum_dict[
                'best_metric_stage1']

            self.best_metric_stage2 = dynamic_momentum_dict[
                'best_metric_stage2']
            self.ema_model.momentum = dynamic_momentum_dict['momentum']

            print_log(
                f'Resuming EMA dynamic momentum with: '
                f'restart_epoch={self.restart_epoch}, '
                f'interval={self.interval}, metric={self.metric}, '
                f'cur_metric={self.cur_metric}, '
                f'best_epoch={self.best_epoch}, '
                f'best_metric={self.best_metric}, '
                f'best_epoch_stage1={self.best_epoch_stage1}, '
                f'best_metric_stage1={self.best_metric_stage1}, '
                f'best_metric_stage2={self.best_metric_stage2}, '
                f'momentum={self.ema_model.momentum}',
                logger=runner.logger)

    def after_load_checkpoint(self, runner: Runner, checkpoint: Dict) -> None:
        self._load_ema_dynamic_momentum_dict(runner, checkpoint)
        super().after_load_checkpoint(runner, checkpoint)
