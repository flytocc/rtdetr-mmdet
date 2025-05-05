# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from torch import nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import MODELS

from mmdet.registry import HOOKS


@HOOKS.register_module()
class DataPreprocessorSwitchHook(Hook):
    """Switch the data_preprocessor during training.

    Args:
        switch_epoch (int): switch pipeline at this epoch.
        switch_data_preprocessor (list[dict]): the data_preprocessor to switch to.
    """

    def __init__(
        self,
        switch_epoch,
        switch_data_preprocessor: Optional[Union[dict, nn.Module]] = None,
    ) -> None:
        self.switch_epoch = switch_epoch
        if switch_data_preprocessor is None:
            switch_data_preprocessor = dict(type='BaseDataPreprocessor')
        if isinstance(switch_data_preprocessor, nn.Module):
            self.switch_data_preprocessor = switch_data_preprocessor
        elif isinstance(switch_data_preprocessor, dict):
            self.switch_data_preprocessor = MODELS.build(
                switch_data_preprocessor)
        else:
            raise TypeError('switch_data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(switch_data_preprocessor)}')
        self._has_switched = False

    def before_train_epoch(self, runner) -> None:
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        model = runner.model
        # TODO: refactor after mmengine using model wrapper
        if is_model_wrapper(model):
            model = model.module
        if epoch >= self.switch_epoch and not self._has_switched:
            runner.logger.info('Switch data_preprocessor now!')
            model.data_preprocessor = self.switch_data_preprocessor.to(
                model.data_preprocessor.device)
            self._has_switched = True
