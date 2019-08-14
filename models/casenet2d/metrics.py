import importlib

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure

from models.casenet2d.losses import compute_per_channel_dice, expand_as_one_hot
from utils.helper import get_logger, adapted_rand
import warnings
warnings.filterwarnings("ignore")

logger = get_logger('EvalMetric')

SUPPORTED_METRICS = ['DiceCoefficient', 'MeanIoU', 'PrecisionStats', 'STEALEdgeLoss']


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, skip_channels=(), epsilon=1e-10, ignore_index=None, **kwargs):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: Soft Dice Coefficient averaged over all channels/classes
        """
        # Average across channels in order to get the final score
        n_classes = input.size()[1]
        if target.dim() < input.dim():
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index))

    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 5

        n_classes = input.size()[1]
        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        # batch dim must be 1
        input = input[0]
        target = target[0]
        assert input.size() == target.size()

        binary_prediction = self._binarize_predictions(input, n_classes)

        if self.ignore_index is not None:
            # zero out ignore_index
            mask = target == self.ignore_index
            binary_prediction[mask] = 0
            target[mask] = 0

        # convert to uint8 just in case
        binary_prediction = binary_prediction.byte()
        target = target.byte()

        per_channel_iou = []
        for c in range(n_classes):
            if c in self.skip_channels:
                continue

            per_channel_iou.append(self._jaccard_index(binary_prediction[c], target[c]))

        assert per_channel_iou, "All channels were ignored from the computation"
        return torch.mean(torch.tensor(per_channel_iou))

    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target, eps=1e-5):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / (torch.sum(prediction | target).float() + eps)

class PrecisionStats:
    """
    Computes Average Precision, Recall, F score given prediction and ground truth instance segmentation.
    """

    def __init__(self, nthresh = 9, ignore_index = None, epsilon = 1e-10, **kwargs):
        """
        :param nthresh: number of points in PR curve
        :param ignore_index: label to be ignored during computation
        """
        self.threshold = np.linspace(1/(nthresh+1), 1 - 1/(nthresh+1), nthresh)
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def __call__(self, input, target):
        """
        :param input: 4D probability maps torch float tensor (NxCxHxW)
        :param target: 3D ground truth instance segmentation torch long tensor (NxHxW)
        :return: average precision statistics among each channel
        """
        n_classes = input.size()[1]
        if target.dim() < input.dim():
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)
        input = input.detach().cpu().numpy()
        target = target.detach().cpu().numpy().astype(np.bool)

        precisions = []
        recalls = []
        fms = []
        for thresh in self.threshold:
            prediction = input > thresh
            tp = np.logical_and(prediction, target).sum(axis=2).sum(axis=2)
            fp = np.logical_and(prediction, ~target).sum(axis=2).sum(axis=2)
            fn = np.logical_and(~prediction, target).sum(axis=2).sum(axis=2)

            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            fm = 2 * precision * recall / (precision + recall + self.epsilon)

            bdryExist = (target.sum(axis=2).sum(axis=2)) > 0

            precision[~bdryExist] = None
            recall[~bdryExist] = None
            fm[~bdryExist] = None

            precisions.append(precision)
            recalls.append(recall)
            fms.append(fm)
        
        AP = np.nanmax(np.nanmean(precisions, axis=1), axis=0)
        AR = np.nanmax(np.nanmean(recalls, axis=1), axis=0)
        AFM = np.nanmax(np.nanmean(fms, axis=1), axis=0)
        return AP, AR, AFM

class STEALEdgeLoss:

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        self.weight = weight
        self.ignore_index = ignore_index

    def __call__(self, input, target):
        """
        Computes STEAL edge loss
        :param input: 4D input tensor (NCHW)
        :param target: 3D target tensor (NHW)
        :return: STEALEdgeLoss
        """
        n_classes = input.size()[1]
        if target.dim() < input.dim():
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)
        weight_sum = target.sum(dim=1).sum(dim=1).sum(dim=1)
        edge_weight = weight_sum.float() / (target.size()[2] * target.size()[3])
        edge_weight = edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        non_edge_weight = 1 - edge_weight

        one_sigmoid_out = torch.sigmoid(input)
        zero_sigmoid_out = 1 - one_sigmoid_out

        loss = - non_edge_weight * target.float() * torch.log(one_sigmoid_out.clamp(min = 1e-10)) -  edge_weight * (1 - target.float()) * torch.log(zero_sigmoid_out.clamp(min = 1e-10))

        return (loss.mean(dim = 0)).sum()

def get_evaluation_metric(config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    def _metric_class(class_name):
        m = importlib.import_module('models.casenet2d.metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)
