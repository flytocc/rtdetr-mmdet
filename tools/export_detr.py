import argparse
from copy import deepcopy
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from mmengine.config import Config, DictAction
from mmengine.dataset import pseudo_collate
from mmengine.model import BaseDataPreprocessor
from mmdet.utils.setup_env import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet export a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def process_model_config(model_cfg: Config,
                         imgs: Union[Sequence[str], Sequence[np.ndarray]],
                         input_shape: Optional[Sequence[int]] = None):
    """Process the model config.

    Args:
        model_cfg (Config): The model config.
        imgs (Sequence[str] | Sequence[np.ndarray]): Input image(s), accepted
            data type are List[str], List[np.ndarray].
        input_shape (list[int]): A list of two integer in (width, height)
            format specifying input shape. Default: None.

    Returns:
        Config: the model config after processing.
    """

    cfg = model_cfg.copy()

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

    pipeline = cfg.test_pipeline

    for i, transform in enumerate(pipeline):
        # for static exporting
        if input_shape is not None:
            if transform.type == 'Resize':
                pipeline[i].keep_ratio = False
                pipeline[i].scale = tuple(input_shape)
            elif transform.type in ('YOLOv5KeepRatioResize', 'LetterResize'):
                pipeline[i].scale = tuple(input_shape)
            elif transform.type == 'Pad' and 'size' in transform:
                pipeline[i].size = tuple(input_shape)

    pipeline = [
        transform for transform in pipeline
        if transform.type != 'LoadAnnotations'
    ]
    cfg.test_pipeline = pipeline
    return cfg


def create_input(
    model_cfg: Config,
    imgs: Union[str, np.ndarray],
    input_shape: Sequence[int] = None,
    data_preprocessor: Optional[BaseDataPreprocessor] = None
) -> Tuple[Dict, torch.Tensor]:
    """Create input for detector.

    Args:
        model_cfg (Config): The model config.
        imgs (str|np.ndarray): Input image(s), accpeted data type are
            `str`, `np.ndarray`.
        input_shape (list[int]): A list of two integer in (width, height)
            format specifying input shape. Defaults to `None`.
        data_preprocessor (BaseDataPreprocessor): The data preprocessor
            of the model. Default to `None`.

    Returns:
        tuple: (data, img), meta information for the input image and input.
    """

    from mmcv.transforms import Compose
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    cfg = process_model_config(model_cfg, imgs, input_shape)
    # Drop pad_to_square when static shape. Because static shape should
    # ensure the shape before input image.

    pipeline = cfg.test_pipeline
    transform = pipeline[1]
    if 'transforms' in transform:
        transform_list = transform['transforms']
        for i, step in enumerate(transform_list):
            if step['type'] == 'Pad' and 'pad_to_square' in step \
                and step['pad_to_square']:
                transform_list.pop(i)
                break
    test_pipeline = Compose(pipeline)
    data = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)
        # build the data pipeline
        data_ = test_pipeline(data_)
        data.append(data_)

    data = pseudo_collate(data)
    if data_preprocessor is not None:
        data = data_preprocessor(data, False)

    return data, data['inputs']


def build_pytorch_model(model_cfg: Config,
                        model_checkpoint: Optional[str] = None,
                        device: Union[str, torch.device] = 'cpu',
                        **kwargs) -> torch.nn.Module:
    """Initialize torch model.

    Args:
        model_cfg (Config): The model config.
        model_checkpoint (str): The checkpoint file of torch model,
            defaults to `None`.

    Returns:
        nn.Module: An initialized torch model generated by other OpenMMLab
            codebases.
    """
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import MODELS

    model = deepcopy(model_cfg.model)
    model.pop('pretrained', None)
    preprocess_cfg = deepcopy(model_cfg.get('preprocess_cfg', {}))
    preprocess_cfg.update(
        deepcopy(model_cfg.get('data_preprocessor', {})))
    model.setdefault('data_preprocessor', preprocess_cfg)
    model = MODELS.build(model)
    if model_checkpoint is not None:
        from mmengine.runner.checkpoint import load_checkpoint
        load_checkpoint(model, model_checkpoint, map_location=device)

    model = revert_sync_batchnorm(model)
    if hasattr(model, 'backbone') and hasattr(model.backbone,
                                              'switch_to_deploy'):
        model.backbone.switch_to_deploy()

    if hasattr(model, 'switch_to_deploy') and callable(
            model.switch_to_deploy):
        model.switch_to_deploy()

    model = model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    register_all_modules(init_default_scope=True)

    # build the model from config
    model = build_pytorch_model(cfg, args.checkpoint, args.device)
    # print(model)

    img = "demo/demo.jpg"
    data, model_inputs = create_input(
        cfg,
        img,
        input_shape=None,
        data_preprocessor=getattr(model, 'data_preprocessor', None))

    if isinstance(model_inputs, list) and len(model_inputs) == 1:
        model_inputs = model_inputs[0]
    data_samples = data['data_samples']

    img_shape = model_inputs.shape[-2:]
    for data_sample in data_samples:
        data_sample.set_field(
            name='img_shape', value=img_shape, field_type='metainfo')

    def __predict_impl(self, batch_inputs, **kwargs):
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats, data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict, rescale=False, batch_data_samples=data_samples)
        bboxes = torch.stack([results.bboxes for results in results_list])
        scores = torch.stack([results.scores for results in results_list])
        labels = torch.stack([results.labels for results in results_list])
        dets = torch.cat((bboxes, scores.unsqueeze(-1)), dim=-1)
        return dets, labels

    from types import MethodType
    model.forward = MethodType(__predict_impl, model)

    # force to export on cpu
    model = model.cpu()
    if isinstance(model_inputs, torch.Tensor):
        model_inputs = model_inputs.cpu()
    elif isinstance(model_inputs, (tuple, list)):
        model_inputs = tuple([_.cpu() for _ in model_inputs])
    else:
        raise RuntimeError(f'Not supported args: {model_inputs}')

    # start export
    torch.onnx.export(
        model,
        model_inputs,
        args.out,
        # export_params=False,
        input_names=['input'],
        output_names=['dets', 'labels'],
        dynamic_axes=None,
        opset_version=17)


if __name__ == '__main__':
    main()
