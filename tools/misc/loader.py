import copy
import logging
from functools import partial
from typing import Any, Dict, Generator, Mapping, Optional, Sequence, Union

import torch
from mmcv.transforms import Compose
from mmengine.dataset import worker_init_fn as default_worker_init_fn
from mmengine.dist import get_rank, get_world_size
from mmengine.logging import print_log
from mmengine.registry import DATA_SAMPLERS, DATASETS, FUNCTIONS
from mmengine.runner.runner import _SlicedDataset
from mmengine.runner.utils import _get_batch_size
from mmengine.structures import BaseDataElement
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from torch.utils.data import DataLoader

from mmdet.engine import PipelineSwitchHook as OriPipelineSwitchHook
from mmdet.registry import HOOKS

CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str,
                 None]


class PrefetchLoader:

    def __init__(self,
                 loader: DataLoader,
                 non_blocking: Optional[bool] = False) -> None:
        assert torch.cuda.is_available()
        self.loader = loader
        self.device = torch.device('cuda')
        self._non_blocking = non_blocking

    def cast_data(self, data: CastData) -> CastData:
        """Copying data to the target device.

        Args:
            data (dict): Data returned by ``DataLoader``.

        Returns:
            CollatedResult: Inputs and data sample at target device.
        """
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, tuple) and hasattr(data, '_fields'):
            # namedtuple
            return type(data)(*(self.cast_data(sample) for sample in data))  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, (torch.Tensor, BaseDataElement)):
            return data.to(self.device, non_blocking=self._non_blocking)
        else:
            return data

    def __iter__(self) -> Generator[CastData, Any, None]:
        first = True
        stream = torch.cuda.Stream()
        stream_context = torch.cuda.stream(stream)

        for next_data in self.loader:

            with stream_context:
                next_data = self.cast_data(next_data)

            if not first:
                yield data  # noqa
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)

            data = next_data

        yield data

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) \
            if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


@HOOKS.register_module(force=True)
class PipelineSwitchHook(OriPipelineSwitchHook):

    def before_train_epoch(self, runner):
        """switch pipeline."""
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        if isinstance(train_loader, PrefetchLoader):
            train_loader = train_loader.loader
        if epoch >= self.switch_epoch and not self._has_switched:
            runner.logger.info(
                f'Switch pipeline after epoch: {self.switch_epoch}!')
            # The dataset pipeline cannot be updated when
            # persistent_workers is True, so we need to force
            # the dataloader's multi-process restart.
            # This is a very hacky approach.
            train_loader.dataset.pipeline = Compose(self.switch_pipeline)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            if isinstance(train_loader, MultiEpochsDataLoader):
                train_loader.iterator = super(MultiEpochsDataLoader,
                                              train_loader).__iter__()
            self._has_switched = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True


def build_dataloader(dataloader: Union[DataLoader, Dict],
                     seed: Optional[int] = None,
                     diff_rank_seed: bool = False) -> DataLoader:
    """Build dataloader.

    The method builds three components:

    - Dataset
    - Sampler
    - Dataloader

    An example of ``dataloader``::

        dataloader = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_size=1,
            num_workers=9
        )

    Args:
        dataloader (DataLoader or dict): A Dataloader object or a dict to
            build Dataloader object. If ``dataloader`` is a Dataloader
            object, just returns itself.
        seed (int, optional): Random seed. Defaults to None.
        diff_rank_seed (bool): Whether or not set different seeds to
            different ranks. If True, the seed passed to sampler is set
            to None, in order to synchronize the seeds used in samplers
            across different ranks.


    Returns:
        Dataloader: DataLoader build from ``dataloader_cfg``.
    """
    if isinstance(dataloader, DataLoader):
        return dataloader

    dataloader_cfg = copy.deepcopy(dataloader)

    # build dataset
    dataset_cfg = dataloader_cfg.pop('dataset')
    if isinstance(dataset_cfg, dict):
        dataset = DATASETS.build(dataset_cfg)
        if hasattr(dataset, 'full_init'):
            dataset.full_init()
    else:
        # fallback to raise error in dataloader
        # if `dataset_cfg` is not a valid type
        dataset = dataset_cfg

    num_batch_per_epoch = dataloader_cfg.pop('num_batch_per_epoch', None)
    if num_batch_per_epoch is not None:
        world_size = get_world_size()
        num_samples = (
            num_batch_per_epoch * _get_batch_size(dataloader_cfg) * world_size)
        dataset = _SlicedDataset(dataset, num_samples)

    # build sampler
    sampler_cfg = dataloader_cfg.pop('sampler')
    if isinstance(sampler_cfg, dict):
        sampler_seed = None if diff_rank_seed else seed
        sampler = DATA_SAMPLERS.build(
            sampler_cfg, default_args=dict(dataset=dataset, seed=sampler_seed))
    else:
        # fallback to raise error in dataloader
        # if `sampler_cfg` is not a valid type
        sampler = sampler_cfg

    # build batch sampler
    batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
    if batch_sampler_cfg is None:
        batch_sampler = None
    elif isinstance(batch_sampler_cfg, dict):
        batch_sampler = DATA_SAMPLERS.build(
            batch_sampler_cfg,
            default_args=dict(
                sampler=sampler, batch_size=dataloader_cfg.pop('batch_size')))
    else:
        # fallback to raise error in dataloader
        # if `batch_sampler_cfg` is not a valid type
        batch_sampler = batch_sampler_cfg

    # build dataloader
    init_fn: Optional[partial]

    if 'worker_init_fn' in dataloader_cfg:
        worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
        worker_init_fn_type = worker_init_fn_cfg.pop('type')
        if isinstance(worker_init_fn_type, str):
            worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
        elif callable(worker_init_fn_type):
            worker_init_fn = worker_init_fn_type
        else:
            raise TypeError(
                'type of worker_init_fn should be string or callable '
                f'object, but got {type(worker_init_fn_type)}')
        assert callable(worker_init_fn)
        init_fn = partial(worker_init_fn, **worker_init_fn_cfg)  # type: ignore
    else:
        if seed is not None:
            disable_subprocess_warning = dataloader_cfg.pop(
                'disable_subprocess_warning', False)
            assert isinstance(
                disable_subprocess_warning,
                bool), ('disable_subprocess_warning should be a bool, but got '
                        f'{type(disable_subprocess_warning)}')
            init_fn = partial(
                default_worker_init_fn,
                num_workers=dataloader_cfg.get('num_workers'),
                rank=get_rank(),
                seed=seed,
                disable_subprocess_warning=disable_subprocess_warning)
        else:
            init_fn = None

    # `persistent_workers` requires pytorch version >= 1.7
    if ('persistent_workers' in dataloader_cfg
            and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
        print_log(
            '`persistent_workers` is only available when '
            'pytorch version >= 1.7',
            logger='current',
            level=logging.WARNING)
        dataloader_cfg.pop('persistent_workers')

    # The default behavior of `collat_fn` in dataloader is to
    # merge a list of samples to form a mini-batch of Tensor(s).
    # However, in mmengine, if `collate_fn` is not defined in
    # dataloader_cfg, `pseudo_collate` will only convert the list of
    # samples into a dict without stacking the batch tensor.
    collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                        dict(type='pseudo_collate'))
    if isinstance(collate_fn_cfg, dict):
        collate_fn_type = collate_fn_cfg.pop('type')
        if isinstance(collate_fn_type, str):
            collate_fn = FUNCTIONS.get(collate_fn_type)
        else:
            collate_fn = collate_fn_type
        collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
    elif callable(collate_fn_cfg):
        collate_fn = collate_fn_cfg
    else:
        raise TypeError(
            'collate_fn should be a dict or callable object, but got '
            f'{collate_fn_cfg}')

    use_prefetcher = dataloader_cfg.pop('use_prefetcher', False)
    use_multi_epochs_loader = dataloader_cfg.pop('use_multi_epochs_loader',
                                                 False)
    loader_class = MultiEpochsDataLoader \
        if use_multi_epochs_loader else DataLoader

    data_loader = loader_class(
        dataset=dataset,
        sampler=sampler if batch_sampler is None else None,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        worker_init_fn=init_fn,
        **dataloader_cfg)
    if use_prefetcher:
        data_loader = PrefetchLoader(data_loader)
    return data_loader
