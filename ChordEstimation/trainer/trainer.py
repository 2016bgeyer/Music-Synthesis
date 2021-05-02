import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer


class Trainer(BaseTrainer):
    
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target, extra=None):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target, extra=extra)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target, extra) in enumerate(self.data_loader):
            for k in data.keys():
                data[k] = data[k].to(self.device)
            for k in target.keys():
                target[k] = target[k].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data, extra=extra)
            loss = self.loss(output, target, extra=extra)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()

            for k in output.keys():
                output[k] = output[k].cpu()
            for k in target.keys():
                target[k] = target[k].cpu()
            total_metrics += self._eval_metrics(output, target, extra=extra)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation and len(self.valid_data_loader) > 0:
            valid_log = self._valid_epoch(epoch)
            log = {**log, **valid_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'valid_metrics'.
        """
        self.model.eval()
        total_valid_loss = 0
        total_valid_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target, extra) in enumerate(self.valid_data_loader):
                for k in data.keys():
                    data[k] = data[k].to(self.device)
                for k in target.keys():
                    target[k] = target[k].to(self.device)

                output = self.model(data, extra=extra)
                loss = self.loss(output, target, extra)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_valid_loss += loss.item()

                for k in output.keys():
                    output[k] = output[k].cpu()
                for k in target.keys():
                    target[k] = target[k].cpu()
                total_valid_metrics += self._eval_metrics(output, target, extra=extra)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        return {
            'valid_loss': total_valid_loss / len(self.valid_data_loader),
            'valid_metrics': (total_valid_metrics / len(self.valid_data_loader)).tolist()
        }
