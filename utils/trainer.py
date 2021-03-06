import logging
import os
import pdb

from tqdm import tqdm 
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.helper import RunningAverage, save_checkpoint, load_checkpoint, get_logger

from utils.contours import ContourBox
import nibabel as nib

class NNTrainer:
    """Network trainer.

    Args:
        model: network model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metricc
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        align_start_iters (int): number of iterations before alignment start
        align_after_iters (int): number of iterations between two alignment steps
        level_set_config (dict): configure files for level set alignment
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs=100, max_num_iterations=None,
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 align_start_iters = None, align_after_iters = None,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 level_set_config = None,
                 logger=None):
        if logger is None:
            self.logger = get_logger('UNet3DTrainer', level=logging.DEBUG)
        else:
            self.logger = logger

        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.align_start_iters = align_start_iters
        self.align_after_iters = align_after_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.level_set_config = level_set_config
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, device, loaders,
                        align_start_iters = None, align_after_iters = None, level_set_config = None, logger=None):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, loaders, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   align_start_iters = align_start_iters,
                   align_after_iters = align_after_iters,
                   level_set_config = level_set_config,
                   logger=logger)

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, loaders,
                        max_num_epochs=100, max_num_iterations=1e5,
                        validate_after_iters=100, log_after_iters=100,
                        validate_iters=None, num_iterations=1, num_epoch=0,
                        align_start_iters = None, align_after_iters = None,
                        eval_score_higher_is_better=True, best_eval_score=None,
                        level_set_config = None,
                        logger=None):
        logger.info(f"Logging pre-trained model from '{pre_trained}'...")
        load_checkpoint(pre_trained, model, None)
        checkpoint_dir = os.path.split(pre_trained)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, loaders, checkpoint_dir,
                   eval_score_higher_is_better=eval_score_higher_is_better,
                   best_eval_score=best_eval_score,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   max_num_iterations=max_num_iterations,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   validate_iters=validate_iters,
                   align_start_iters = align_start_iters,
                   align_after_iters = align_after_iters,
                   level_set_config = level_set_config,
                   logger=logger)

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'])

            if should_terminate:
                break

            self.num_epoch += 1

    def train(self, train_loader):
        """Trains the model for 1 epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = RunningAverage()
        train_eval_scores = RunningAverage()
        self.logger.info(
            f'Training epoch [{self.num_epoch}/{self.max_num_epochs - 1}], iteration per epoch: {len(train_loader)}. ')
        # sets the model in training mode
        self.model.train()
        if self.validate_after_iters is None:
            self.validate_after_iters = len(train_loader)
        if self.log_after_iters is None:
            self.log_after_iters = 1
        if self.max_num_iterations is None:
            self.max_num_iterations = self.max_num_epochs * len(train_loader)
        if self.align_start_iters is None:
            self.align_start_iters = self.max_num_iterations
        if self.align_after_iters is None:
            self.align_after_iters = self.max_num_iterations

        for i, t in enumerate(train_loader):
            input, target, weight = self._split_training_batch(t)
            output, loss = self._forward_pass(input, target, weight)
            train_losses.update(loss.item(), self._batch_size(input))

            # if model contains final_activation layer for normalizing logits apply it, otherwise both
            # the evaluation metric as well as images in tensorboard will be incorrectly computed
            if hasattr(self.model, 'final_activation'):
                if self.model.final_activation is not None:
                    output = self.model.final_activation(output)

            # compute eval criterion
            eval_score = self.eval_criterion(output, target)
            train_eval_scores.update(eval_score.item(), self._batch_size(input))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (self.num_iterations == 1) or (self.num_iterations % self.log_after_iters == 0):
                # log stats, params and images
                self.logger.info(
                    f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. Batch [{i}/{len(train_loader) - 1}]. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')
                self.logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_stats('train', train_losses.avg, train_eval_scores.avg)

                train_losses = RunningAverage()
                train_eval_scores = RunningAverage()

            if (self.num_iterations == 1) or (self.num_iterations % self.validate_after_iters == 0):
                # evaluate on validation set
                eval_score = self.validate(self.loaders['val'])
                # adjust learning rate if necessary
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_score)
                else:
                    self.scheduler.step()
                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)
                # save checkpoint
                self._save_checkpoint(is_best)
                self._log_params()
                #self._log_images(input, target, output)

            if (self.num_iterations >= self.align_start_iters) and ((self.num_iterations - self.align_start_iters) % self.align_after_iters == 0):
                self.loaders['train'] = self.align(self.loaders['train'])

            if self.max_num_iterations < self.num_iterations:
                self.logger.info(
                    f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                return True

            self.num_iterations += 1
            
        return False

    def validate(self, val_loader):

        val_losses = RunningAverage()
        val_scores = RunningAverage()
        
        self.logger.info(f'Validating epoch [{self.num_epoch}/{self.max_num_epochs - 1}]. ')
        if self.validate_iters is None:
            self.validate_iters = len(val_loader)
        val_iterator = iter(val_loader)
        
        try:
            # set the model in evaluation mode; final_activation doesn't need to be called explicitly
            self.model.eval()
            with torch.no_grad():
                for i in tqdm(range(self.validate_iters)):
                    try:
                        batch = next(val_iterator)
                        input, target, weight = self._split_training_batch(batch)
                    except StopIteration:
                        val_iterator = iter(val_loader)
                        batch = next(val_iterator)
                        input, target, weight = self._split_training_batch(batch)

                    output, loss = self._forward_pass(input, target, weight)
                    val_losses.update(loss.item(), self._batch_size(input))

                    eval_score = self.eval_criterion(output, target)
                    val_scores.update(eval_score.item(), self._batch_size(input))

                self._log_stats('val', val_losses.avg, val_scores.avg)
                self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
                return val_scores.avg
        finally:
            # set back in training mode
            self.model.train()

    def align(self, loader):
        self.logger.info(f'Level set alignment at [{self.num_iterations}/{self.max_num_iterations}], Epoch [{self.num_epoch}/{self.max_num_epochs - 1}].')
        assert self.level_set_config is not None
        if self.level_set_config['prefix'] is not None:
            folderpath = self.level_set_config['prefix']+str(self.num_iterations)
            if not os.path.exists(folderpath):
                os.mkdir(folderpath)
        dim = self.level_set_config.get('dim', 2)
        n_workers = self.level_set_config.get('n_workers', 0)
        cbox = ContourBox.LevelSetAlignment(n_workers=n_workers, config=self.level_set_config)
        datasets = loader.dataset.datasets
        try:
            # set the model in evaluation mode; final_activation doesn't need to be called explicitly
            self.model.eval()
            with torch.no_grad():
                for i in tqdm(range(len(datasets))):
                    iz, iy, ix = datasets[i].raw.shape
                    affine = datasets[i].affine
                    if dim == 2:
                        dz = 1
                    elif dim == 3:
                        dz = self.level_set_config.get('dz', 1)
                    batch_size = self.level_set_config.get('batch_size', 1)
                    js = 0
                    while js < iz:
                        je = min(iz, js+dz*batch_size)
                        if (je - js) % dz != 0:
                            je = iz - (je - js) % dz
                            batch_size = 1
                            dz = iz - je
                        idx = slice(js, je)
                        js = je

                        data_sliced = (datasets[i].raw[idx]).astype(np.float32)
                        label_sliced = np.reshape((datasets[i].label[idx]), (-1, dz, iy, ix)).astype(np.long)

                        if dim == 2:
                            input = torch.from_numpy(data_sliced[:,np.newaxis,:,:]).to(self.device)
                            pred = torch.sigmoid(self.model(input))
                            pred = pred.unsqueeze(2)

                        elif dim == 3:
                            data_sliced = np.reshape(data_sliced, (-1, dz, iy, ix)).astype(np.float32)
                            input = torch.from_numpy(data_sliced[:,np.newaxis,:,:,:]).to(self.device)
                            pred = torch.sigmoid(self.model(input)).squeeze(0)
                            
                        gt = self._expand_as_one_hot(torch.from_numpy(label_sliced).to(self.device), pred.shape[1])
                        output = cbox({'seg': gt, 'bdry': None}, pred)
                        output = np.multiply(np.sum(output, axis=1) > 0, np.argmax(output, axis=1) + 1)
                        loader.dataset.datasets[i].label[idx] = np.reshape(output, (-1, iy, ix))
                    if self.level_set_config['prefix'] is not None:
                        output_file = self._get_output_file(datasets[i], folderpath=folderpath, suffix='_refine')
                        nib.save(nib.Nifti1Image((np.transpose(loader.dataset.datasets[i].label).astype(np.int16)), affine), output_file)
                return loader
        finally:
            # set back in training mode
            self.model.train()

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight

    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output = self.model(input)

        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        if torch.cuda.device_count() > 1:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': model_state,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self._images_from_batch(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='HW')

    def _images_from_batch(self, name, batch):
        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))
        else:
            # batch has no channel or dim: NDHW
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img, eps = 1e-5):
        return (img - np.min(img)) / (np.ptp(img) + eps)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

    @staticmethod
    def _get_output_file(dataset, folderpath = None, suffix='_predictions', ext = 'nii.gz'):
        filename = (os.path.basename(dataset.file_path)).split('.')[0]
        if folderpath is None:
            folderpath = os.path.dirname(dataset.file_path)
        return f'{os.path.join(folderpath, filename)}{suffix}.{ext}'

    @staticmethod
    def _expand_as_one_hot(input, C):
        shape = input.size()
        shape = list(shape)
        shape.insert(1, C+1)
        shape = tuple(shape)

        # expand the input tensor to NxCx(D)xHxW
        src = input.unsqueeze(1)
        if input.dim() == 3:
            return torch.zeros(shape).to(input.device).scatter_(1, src, 1)[:, 1:, :, :]
        elif input.dim() == 4:
            return torch.zeros(shape).to(input.device).scatter_(1, src, 1)[:, 1:, :, :, :]