import importlib
import argparse
import os

import numpy as np
import torch
import nibabel as nib

from utils.helper import get_logger, load_checkpoint, unpad, RunningAverage
from utils.config import load_config
from models.casenet2d.model import get_model
from models.casenet2d.metrics import get_evaluation_metric

from utils.databuilder import get_test_loaders


def predict(model, data_loader, output_file, config, logger):
    """
    Return prediction masks by applying the model on the given dataset

    Args:
        model: trained model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        output_file (str): path to the output file
        config (dict): global config dict

    Returns:
         prediction_maps (numpy array): prediction masks for given dataset
    """

    def _volume_shape(dataset):
        raw = dataset.raw
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    out_channels = config['model'].get('out_channels')
    assert out_channels is not None

    # store evaluation metrics
    eval_scores = []

    prediction_channel = config.get('prediction_channel', None)
    if prediction_channel is not None:
        logger.info(f"Using only channel '{prediction_channel}' from the network output")

    device = config['device']

    logger.info(f'Running prediction on {len(data_loader)} patches...')
    # dimensionality of the the output (CxDxHxW)
    volume_shape = _volume_shape(data_loader.dataset)
    if prediction_channel is None:
        prediction_maps_shape = (out_channels,) + volume_shape
    else:
        # single channel prediction map
        prediction_maps_shape = (1,) + volume_shape

    logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

    # initialize the output prediction arrays
    prediction_map = np.zeros(prediction_maps_shape, dtype='float32')
    # initialize normalization mask in order to average out probabilities of overlapping patches
    normalization_mask = np.zeros(prediction_maps_shape, dtype='float32')

    # Sets the module in evaluation mode explicitly, otherwise the final Softmax/Sigmoid won't be applied!
    model.eval()
    # Run predictions on the entire input dataset
    with torch.no_grad():
        for patch, index, target, t_index in data_loader:
            #logger.info(f'Predicting slice:{index}')

            # save patch index: (C,D,H,W)
            if prediction_channel is None:
                channel_slice = slice(0, out_channels)
            else:
                channel_slice = slice(0, 1)

            index = (channel_slice,) + tuple(index)

            # send patch to device
            patch = patch.to(device)
            # forward pass
            prediction = model(patch)

            if target is not None:
                eval_criterion = get_evaluation_metric(config)
                eval_score = eval_criterion(prediction, target)
                eval_scores.append(eval_score)
                #logger.info(f'Current evaluation score: {np.nanmax(eval_score, axis = 1)}.')

            # squeeze batch dimension and convert back to numpy array
            prediction = prediction.squeeze(dim=0).cpu().numpy()
            if prediction_channel is not None:
                # use only the 'prediction_channel'
                #logger.info(f"Using channel '{prediction_channel}'...")
                prediction = np.expand_dims(prediction[prediction_channel], axis=0)

            # unpad in order to avoid block artifacts in the output probability maps
            u_prediction, u_index = unpad(prediction, index, volume_shape)
            # accumulate probabilities into the output prediction array
            prediction_map[u_index] += u_prediction
            # count voxel visits for normalization
            normalization_mask[u_index] += 1

    # save probability maps
    prediction_map = prediction_map / normalization_mask
    affine = data_loader.dataset.affine
    prediction_map_save = np.transpose(np.floor(prediction_map * 1e4).astype(np.int16))
    nib.save(nib.Nifti1Image(prediction_map_save, affine), output_file)
    return np.nanmean(eval_scores, axis = 0), len(eval_scores)

def _get_output_file(dataset, folderpath = None, suffix='_predictions', ext = 'nii.gz'):
    filename = (os.path.basename(dataset.file_path)).split('.')[0]
    if folderpath is None:
        folderpath = os.path.dirname(dataset.file_path)
    return f'{os.path.join(folderpath, filename)}{suffix}.{ext}'

def get_batch_size(input):
    if isinstance(input, list) or isinstance(input, tuple):
        return input[0].size(0)
    else:
        return input.size(0)

def main():
    # Create main logger
    logger = get_logger('CASENetPredictor')

    parser = argparse.ArgumentParser(description='CASENet2D testing')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', default='/home/SENSETIME/shenrui/Dropbox/SenseTime/edgeDL/resources/test_config_backup.yaml')
    args = parser.parse_args()

    # Load and log experiment configuration
    config = load_config(args.config)
    logger.info(config)

    # Create the model
    model = get_model(config)

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    load_checkpoint(model_path, model)
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(config['device'])
    folderpath = config['save_path']
    logger.info(f'Destination of predictions is {folderpath}...')

    logger.info('Loading datasets...')

    eval_score_avg = 0
    count = 0
    for test_loader in get_test_loaders(config):
        logger.info(f"Processing '{test_loader.dataset.file_path}'...")

        output_file = _get_output_file(test_loader.dataset, folderpath=folderpath)
        # run the model prediction on the entire dataset and save to nifti image
        eval_score, num = predict(model, test_loader, output_file, config, logger)
        eval_score_avg = (eval_score_avg * count + eval_score * num) / (count + num)
        count += num
        logger.info(f'Testing finished. Average evaluation score: {eval_score}. Saving predictions to: {output_file}...')
        logger.info(f'Total average evaluation score: {eval_score_avg}')

if __name__ == '__main__':
    main()
