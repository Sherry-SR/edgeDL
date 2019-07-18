import collections
import importlib

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import os

from models.unet3d.utils import get_logger
from utils import transforms

class SliceBuilder:
    def __init__(self, raw_datasets, patch_shape, stride_shape, label_datasets = None):
        if len(patch_shape) == 2 and len(stride_shape) == 2:
            self._raw_slices = self._build_slices_2D(raw_datasets, patch_shape, stride_shape)
            if label_datasets is None:
                self._label_slices = None
            else:
                # take the first element in the label_datasets to build slices
                self._label_slices = self._build_slices_2D(label_datasets, patch_shape, stride_shape)
                assert len(self._raw_slices) == len(self._label_slices)
        elif len(patch_shape) == 3 and len(stride_shape) == 3:
            self._raw_slices = self._build_slices(raw_datasets, patch_shape, stride_shape)
            if label_datasets is None:
                self._label_slices = None
            else:
                # take the first element in the label_datasets to build slices
                self._label_slices = self._build_slices(label_datasets, patch_shape, stride_shape)
                assert len(self._raw_slices) == len(self._label_slices)
        else:
            raise ValueError(f"Unsupported patch and stride dimensions '{patch_shape}' and '{stride_shape}'")

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _build_slices_2D(dataset, patch_shape, stride_shape):
        """Iterates over a given 3-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of 2D slices, i.e. [(slice, slice), ...]
        """
        slices = []
        assert dataset.ndim == 3
        i_z, i_y, i_x = dataset.shape

        k_y, k_x = patch_shape
        s_y, s_x = stride_shape
        for z in range(i_z):
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        z,
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

class NiftiDataset(Dataset):
    def __init__(self, file_path, patch_shape, stride_shape, phase, label_path = None, clip_val = None, transformer_config = None, slice_builder_cls = SliceBuilder):
        """
        :param file_path: path to nifti subject data
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :param label_path: tag of nifti label data
        :param clip_value: clip value within a range
        :param transformer_config: data augmentation configuration
        :param slice_builder_cls: slice builder tool
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.file_path = file_path

        # load nifti image data
        assert os.path.exists(file_path)
        niftiimg = nib.load(file_path)
        self.raw = np.transpose(niftiimg.get_fdata())
        self.affine = niftiimg.affine

        if clip_val is not None:
            self.raw[self.raw > clip_val[1]] = clip_val[1]
            self.raw[self.raw < clip_val[0]] = clip_val[0]

        mean, std = self._calculate_mean_std(self.raw)
        self.transformer = transforms.get_transformer(transformer_config, phase, mean=mean, std=std, clip_val=clip_val)
        self.raw_transform = self.transformer.raw_transform()

        if phase != 'test':
            assert os.path.exists(label_path)
            niftiseg = nib.load(label_path)
            self.label = np.transpose(niftiseg.get_fdata())
            self.label_transform = self.transformer.label_transform()
            self._check_dimensionality(self.raw, self.label)
        else:
            self.label = None

        # build slice indices for raw and label data sets
        slice_builder = slice_builder_cls(self.raw, patch_shape, stride_shape, self.label)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices

        self.patch_count = len(self.raw_slices)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]

        # get the raw data patch for a given slice
        raw_transformed = self._transform_datasets(self.raw, raw_idx, self.raw_transform)

        if self.phase == 'test':
            # just return the transformed raw data
            return raw_transformed
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_transformed = self._transform_datasets(self.label, label_idx, self.label_transform)
            # return the transformed raw and label data
            return raw_transformed, label_transformed

    def __len__(self):
        return self.patch_count

    @staticmethod
    def _transform_datasets(dataset, idx, transformer):
        transformed_datasets = []
        # get the data and apply the transformer
        data = np.squeeze(dataset[idx])
        transformed_data = transformer(data)
        transformed_datasets.append(transformed_data)

        # if transformed_datasets is a singleton list return the first element only
        if len(transformed_datasets) == 1:
            return transformed_datasets[0]
        else:
            return transformed_datasets

    @staticmethod
    def _calculate_mean_std(input):
        """
        Compute a mean/std of the raw stack for normalization.
        :return: a tuple of (mean, std) of the raw data
        """
        return input.mean(keepdims=True), input.std(keepdims=True)

    @staticmethod
    def _check_dimensionality(raw, label):
        assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        if raw.ndim == 3:
            raw_shape = raw.shape
        else:
            raw_shape = raw.shape[1:]

        assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        if label.ndim == 3:
            label_shape = label.shape
        else:
            label_shape = label.shape[1:]
        assert raw_shape == label_shape, 'Raw and labels have to be of the same size'

def _get_slice_builder_cls(class_name):
    m = importlib.import_module('utils.databuilder')
    clazz = getattr(m, class_name)
    return clazz

def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger = get_logger('TrainDataset')
    logger.info('Creating training and validation set loaders...')

    # get train and validation files
    train_paths = loaders_config['train_path']
    val_paths = loaders_config['val_path']
    assert isinstance(train_paths, list)
    assert isinstance(val_paths, list)

    # get train/validation patch size and stride
    train_patch = tuple(loaders_config['train_patch'])
    train_stride = tuple(loaders_config['train_stride'])
    val_patch = tuple(loaders_config['val_patch'])
    val_stride = tuple(loaders_config['val_stride'])
    # get clip value
    clip_val = tuple(loaders_config['clip_val'])

    slice_builder_str = loaders_config.get('slice_builder', 'SliceBuilder')
    logger.info(f'Slice builder class: {slice_builder_str}')
    slice_builder_cls = _get_slice_builder_cls(slice_builder_str)

    # create nifti backed training and validation dataset with data augmentation
    train_datasets = []
    for train_path in train_paths:
        assert os.path.exists(train_path)
        try:
            logger.info(f'Loading training set from: {train_path}...')
            with open(train_path) as f:
                for line in f:
                    name, file_path, label_path = line.split()[0:3]
                    logger.info(f'Create training dataset from: {name}...')
                    train_dataset = NiftiDataset(file_path, train_patch, train_stride, phase = 'train',
                                                 label_path = label_path, clip_val = clip_val,
                                                 transformer_config = loaders_config['transformer'],
                                                 slice_builder_cls = slice_builder_cls)
                    train_datasets.append(train_dataset)
        except Exception:
            logger.info(f'Skipping training set: {train_path}', exc_info=True)

    val_datasets = []
    for val_path in val_paths:
        assert os.path.exists(val_path)
        try:
            logger.info(f'Loading validation set from: {val_path}...')
            with open(val_path) as f:
                for line in f:
                    name, file_path, label_path = line.split()[0:3]
                    logger.info(f'Create validation dataset from: {name}...')
                    val_dataset = NiftiDataset(file_path, val_patch, val_stride, phase = 'val',
                                                 label_path = label_path, clip_val = clip_val,
                                                 transformer_config = loaders_config['transformer'],
                                                 slice_builder_cls = slice_builder_cls)
                    val_datasets.append(val_dataset)
        except Exception:
            logger.info(f'Skipping validation set: {val_path}', exc_info=True)

    num_workers = loaders_config.get('num_workers', 1)
    batch_size = loaders_config.get('bach_size', 1)
    logger.info(f'Number of workers for train/val datasets: {num_workers}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    }


def get_test_loaders(config):
    """
    Returns a list of DataLoader, one per each test file.

    :param config: a top level configuration object containing the 'datasets' key
    :return: generator of DataLoader objects
    """

    def my_collate(batch):
        error_msg = "batch must contain tensors or slice; found {}"
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, 0)
        elif isinstance(batch[0], slice):
            return batch[0]
        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [my_collate(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

    logger = get_logger('TestDataset')

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    # get test files
    test_paths = loaders_config['test_path']
    assert isinstance(test_paths, list)
    # get test patch size and stride
    test_patch = tuple(loaders_config['test_patch'])
    test_stride = tuple(loaders_config['test_stride'])
    # get clip value
    clip_val = tuple(loaders_config['clip_val'])
    num_workers = loaders_config.get('num_workers', 1)

    for test_path in test_paths:
        assert os.path.exists(test_path)
        try:
            logger.info(f'Loading testing set from: {test_path}...')
            with open(test_path) as f:
                for line in f:
                    name, file_path = line.split()[0:2]
                    logger.info(f'Create testing dataset from: {name}...')
                    test_dataset = NiftiDataset(file_path, test_patch, test_stride, phase = 'test', clip_val = clip_val)                    # use generator in order to create data loaders lazily one by one
                    yield DataLoader(test_dataset, batch_size=1, num_workers=num_workers, collate_fn=my_collate)
        except Exception:
            logger.info(f'Skipping testing set: {test_path}', exc_info=True)