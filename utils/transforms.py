import importlib
import pdb

import numpy as np
import torch
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import binary_dilation, distance_transform_edt
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from torchvision.transforms import Compose


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 2D(HxW), 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state

    def __call__(self, m):
        assert m.ndim in [2, 3, 4], 'Supports only 2D(HxW) 3D (DxHxW) or 4D (CxDxHxW) images'
        if m.ndim == 2:
            axes = (0, 1)
        else:
            axes = (0, 1, 2)

        for axis in axes:
            if self.random_state.uniform() > 0.5:
                if m.ndim == 2 or m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)
        return m


class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 2D(HxW), 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state

    def __call__(self, m):
        assert m.ndim in [2, 3, 4], 'Supports only 2D(HxW), 3D (DxHxW) or 4D (CxDxHxW) images'

        # pick number of rotations at random
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        if m.ndim == 2:
            m = np.rot90(m, k, (0, 1))
        elif m.ndim == 3:
            m = np.rot90(m, k, (1, 2))
        else:
            channels = [np.rot90(m[c], k, (1, 2)) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m

class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=10, axes=None, mode='constant', order=0, **kwargs):
        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m):
        if self.axes is None:
            if m.ndim == 2:
                axes = [(0, 1)]
            else:
                axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(self.axes, list) and len(self.axes) > 0
            axes = self.axes
        axis = axes[self.random_state.randint(len(axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 2 or m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        else:
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                        in range(m.shape[0])]
            m = np.stack(channels, axis=0)
        return m


class RandomContrast:
    """
        Adjust the brightness of an image by a random factor.
    """

    def __init__(self, random_state, factor=0.5, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        self.factor = factor
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            brightness_factor = self.factor + self.random_state.uniform()
            return np.clip(m * brightness_factor, 0, 1)

        return m


# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=3 when transforming the labels
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order!
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, random_state, spline_order, alpha=15, sigma=3, execution_probability=0.3, **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim == 2 or m.ndim == 3
            if m.ndim == 2:
                dy = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
                dx = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
                y_dim, x_dim = m.shape
                y, x = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing='ij')
                indices = y + dy, x + dx
            else:
                dz = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
                dy = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
                dx = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha

                z_dim, y_dim, x_dim = m.shape
                z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
                indices = z + dz, y + dy, x + dx
            return map_coordinates(m, indices, order=self.spline_order, mode='reflect')
        return m

class Normalize:
    """
    Normalizes a given input tensor to be 0-mean and 1-std.
    mean and std parameter have to be provided explicitly.
    """

    def __init__(self, mean, std, eps=1e-4, **kwargs):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, m):
        return (m - self.mean) / (self.std + self.eps)

class ClipNormalize:
    """
    Normalizes a cliped input tensor to [-1, 1].
    clip range have to be provided explicitly.
    """

    def __init__(self, clip_val, **kwargs):
        self.mean = (clip_val[0] + clip_val[1]) / 2.0
        self.scale = (clip_val[1] - clip_val[0]) / 2.0

    def __call__(self, m):
        return (m - self.mean) / self.scale

class RangeNormalize:
    def __init__(self, max_value=255, **kwargs):
        self.max_value = max_value

    def __call__(self, m):
        return m / self.max_value

class GaussianNoise:
    def __init__(self, random_state, max_sigma, max_value=255, **kwargs):
        self.random_state = random_state
        self.max_sigma = max_sigma
        self.max_value = max_value

    def __call__(self, m):
        # pick std dev from [0; max_sigma]
        std = self.random_state.randint(self.max_sigma)
        gaussian_noise = self.random_state.normal(0, std, m.shape)
        noisy_m = m + gaussian_noise
        return np.clip(noisy_m, 0, self.max_value).astype(m.dtype)

class SegToEdge:
    def __init__(self, out_channels, radius = 1, dtype=np.float32, **kwargs):
        self.dtype = dtype
        self.radius = radius
        self.out_channels = out_channels

    def __call__(self, m):
        """
        :param image: semantic map should be HxWx1 with values 0,1,label_ignores
        :param radius: radius size
        :param label_ignores: values to mask.
        :return: edgemap with boundary computed based on radius
        """

        # we need to pad the borders, to solve problems with dt around the boundaries of the image.
        if len(m.shape) == 3:
            image_pad = np.pad(m, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
        elif len(m.shape) == 2:
            image_pad = np.pad(m, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        else:
            raise NotImplementedError
        edges = []
        for i in range(self.out_channels):
            mask = image_pad == (i+1)
            dist1 = distance_transform_edt(mask)
            dist2 = distance_transform_edt(1.0 - mask)
            dist = dist1 + dist2
            # removing padding, it shouldnt affect result other than if the image is seg to the boundary.
            if len(m.shape) == 3:
                dist = dist[1:-1, 1:-1, 1:-1]
            elif len(m.shape) == 2:
                dist = dist[1:-1, 1:-1]
            assert dist.shape == m.shape
            dist[dist > self.radius] = 0
            dist = (dist > 0).astype(np.uint8)  # just 0 or 1
            edges.append(dist)
        edges = np.array(edges, dtype = self.dtype)
        return edges

class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W) or (H, W)).
    """

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [2, 3, 4], 'Supports only 3D (CxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        if self.expand_dims and (m.ndim == 2 or m.ndim == 3):
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))

class Identity:
    def __call__(self, m):
        return m


def get_transformer(config, phase, mean=None, std=None, clip_val=None):
    if phase == 'val':
        phase = 'test'

    assert phase in config, f'Cannot find transformer config for phase: {phase}'
    phase_config = config[phase]
    return Transformer(phase_config, mean, std, clip_val)


class Transformer:
    def __init__(self, phase_config, mean, std, clip_val):
        self.phase_config = phase_config
        self.config_base = {'mean': mean, 'std': std, 'clip_val':clip_val}
        self.seed = 47

    def raw_transform(self):
        return self._create_transform('raw')

    def label_transform(self):
        return self._create_transform('label')

    @staticmethod
    def _transformer_class(class_name):
        m = importlib.import_module('utils.transforms')
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self, name):
        assert name in self.phase_config, f'Could not find {name} transform'
        return Compose([
            self._create_augmentation(c) for c in self.phase_config[name]
        ])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        config['random_state'] = np.random.RandomState(self.seed)
        aug_class = self._transformer_class(config['name'])
        return aug_class(**config)


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input