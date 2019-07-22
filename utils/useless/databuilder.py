import collections
import importlib

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import os

from utils.helper import get_logger
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

class RandomSliceBuilder:
    def __init__(self, data_shape, patch_shape, num_slicers = 1):
        # Slice builder for 3D (DxHxW) or 4D (CxDxHxW) data
        if len(data_shape) == 4:
            self._channels = data_shape[0]
        elif len(data_shape) == 3:
            self._channels = 1
        else:
            raise ValueError(f"Unsupported data dimensions {data_shape}")
        self._slices = self._build_slices(data_shape, patch_shape, num_slicers)
    @property
    def slices(self):
        return self._slices
    @property
    def channel(self):
        return self._channels
    @staticmethod
    def _build_slices(data_shape, patch_shape, num_slicers):
        """Generate num_slicers random patch slices for data with a certain
        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """

        if len(data_shape) == 4:
            slices = [(slice(0, data_shape[0]),)] * num_slicers
            skip = 1
        else:
            slices = [()] * num_slicers
            skip = 0

        if len(patch_shape) == 2:
            s = np.random.randint(0, data_shape[skip], size = num_slicers)
            for j in range(num_slicers):
                slices[j] = slices[j] + (s[j],)

        for i in range(len(patch_shape)):
            s_r = data_shape[i + skip] - patch_shape[i]
            assert s_r >= 0, 'Sample size has to be bigger than the patch size'
            s = np.random.randint(0, s_r+1, size = num_slicers)
            for j in range(num_slicers):
                slices[j] = slices[j] + (slice(s[j], s[j] + patch_shape[i]),)
        return slices
    
    def _slicers_for_labels(self):
        if self._channels > 1:
            return [x[1:] for x in self._slices]
        else:
            return self._slices            
            
class TrainDataset(Dataset):
    def __init__(self, file_path, patch_shape, phase, label_path, clip_val = None, transformer_config = None, slice_builder_cls = RandomSliceBuilder):
        """
        :param file_path: path to nifti subject data
        :param patch_shape: the shape of the patch DxHxW or HxW for each slice
        :param phase: 'train' for training, 'val' for validation; data augmentation is performed only during training
        :param label_path: path to nifti label data
        :param clip_value: clip value within a range
        :param transformer_config: data augmentation configuration
        :param slice_builder_cls: slice builder tool
        """
        assert phase in ['train', 'val']
        self. phase = phase
        self.clip_val = clip_val
        self.patch_shape = patch_shape
        self.transformer_config = transformer_config

        assert isinstance(file_path, list)
        assert isinstance(label_path, list)
        assert len(file_path) == len(label_path)

        self.file_path = file_path
        self.label_path = label_path
        self.slice_builder_cls = slice_builder_cls

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        
        # load nifti image data
        assert os.path.exists(self.file_path[idx])
        niftiimg = nib.load(self.file_path[idx])
        raw = np.transpose(niftiimg.get_fdata())

        if self.clip_val is not None:
            raw[raw > self.clip_val[1]] = self.clip_val[1]
            raw[raw < self.clip_val[0]] = self.clip_val[0]

        if self.transformer_config is not None:
            mean, std = self._calculate_mean_std(raw)
            transformer = transforms.get_transformer(self.transformer_config, self.phase, mean=mean, std=std, clip_val=self.clip_val)
            raw_transform = transformer.raw_transform()
        else:
            raw_transform = None
        slice_builder = self.slice_builder_cls(data_shape = raw.shape, patch_shape = self.patch_shape, num_slicers = 1)
        raw_slices = slice_builder.slices
        raw_transformed = self._transform_datasets(raw, raw_slices[0], raw_transform)

        assert os.path.exists(self.label_path[idx])
        niftiseg = nib.load(self.label_path[idx])
        label = np.transpose(niftiseg.get_fdata())
        self._check_dimensionality(raw, label)

        if self.transformer_config is not None:
            label_transform = transformer.label_transform()
        else:
            label_transform = None

        label_slices = slice_builder._slicers_for_labels()
        label_transformed = self._transform_datasets(label, label_slices[0], label_transform)
        return raw_transformed, label_transformed
    
    def __len__(self):
        return len(self.file_path)

    @staticmethod
    def _transform_datasets(dataset, idx, transformer):
        transformed_datasets = []
        # get the data and apply the transformer
        data = np.squeeze(dataset[idx])
        if transformer is None:
            transformed_data = data
        else:
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
    val_patch = tuple(loaders_config['val_patch'])
    # get clip value
    clip_val = tuple(loaders_config['clip_val'])

    slice_builder_str = loaders_config.get('slice_builder', 'RandomSliceBuilder')
    logger.info(f'Slice builder class: {slice_builder_str}')
    slice_builder_cls = _get_slice_builder_cls(slice_builder_str)

    # create nifti backed training and validation dataset with data augmentation
    for train_path in train_paths:
        assert os.path.exists(train_path)
        try:
            logger.info(f'Loading training set from: {train_path}...')
            file_path = []
            label_path = []
            with open(train_path) as f:
                for line in f:
                    _, fp, lp = line.split()[0:3]
                    file_path.append(fp)
                    label_path.append(lp)
            train_datasets = TrainDataset(file_path = file_path, patch_shape = train_patch, phase = 'train',
                                         label_path = label_path, clip_val = clip_val,
                                         transformer_config = loaders_config['transformer'],
                                         slice_builder_cls = slice_builder_cls)
        except Exception:
            logger.info(f'Skipping training set: {train_path}', exc_info=True)

    for val_path in val_paths:
        assert os.path.exists(val_path)
        try:
            logger.info(f'Loading validation set from: {val_path}...')
            with open(train_path) as f:
                for line in f:
                    _, fp, lp = line.split()[0:3]
                    file_path.append(fp)
                    label_path.append(lp)
            val_datasets = TrainDataset(file_path = file_path, patch_shape = val_patch, phase = 'val',
                                         label_path = label_path, clip_val = clip_val,
                                         transformer_config = loaders_config['transformer'],
                                         slice_builder_cls = slice_builder_cls)
        except Exception:
            logger.info(f'Skipping validation set: {val_path}', exc_info=True)

    num_workers = loaders_config.get('num_workers', 1)
    batch_size = loaders_config.get('bach_size', 1)
    logger.info(f'Number of workers for train/val datasets: {num_workers}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    }