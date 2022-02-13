"""
callbacks functions used in training process
"""
import logging
import os
import re
import warnings
from collections import deque
from glob import glob
from typing import Dict, Callable, Union
from monty.serialization import dumpfn, loadfn
from pathlib import Path

import numpy as np
import tensorflow.keras.backend as kb
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence

from megnet.utils.metrics import mae, accuracy
from megnet.utils.preprocessing import DummyScaler, Scaler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelCheckpointMAE(Callback):
    """
    Save the best MAE model with target scaler
    """

    def __init__(
        self,
        filepath: str = "./callback/val_mae_{epoch:05d}_{val_mae:.6f}.hdf5",
        monitor: Union[str, Callable] = "val_mae",
        log_monitor: Union[str, Callable] = None, 
        verbose: int = 0,
        save_best_only: bool = True,
        save_weights_only: bool = False,
        val_gen: Sequence = None,
        steps_per_val: int = None,
        target_scaler: Scaler = None,
        period: int = 1,
        mode: str = "auto", 
    ):
        """
        Args:
            filepath (string): path to save the model file with format. For example
                `weights.{epoch:02d}-{val_mae:.6f}.hdf5` will save the corresponding epoch and
                val_mae in the filename
            monitor (string): quantity to monitor, default to "val_mae"
            verbose (int): 0 for no training log, 1 for only epoch-level log and 2 for batch-level log
            save_best_only (bool): whether to save only the best model
            save_weights_only (bool): whether to save the weights only excluding model structure
            val_gen (generator): validation generator
            steps_per_val (int): steps per epoch for validation generator
            target_scaler (object): exposing inverse_transform method to scale the output
            period (int): number of epoch interval for this callback
            mode: (string) choose from "min", "max" or "auto"
        """
        super().__init__()
        if val_gen is None:
            raise ValueError("No validation data is provided!")
        self.verbose = verbose
        if self.verbose > 0:
            logging.basicConfig(level=logging.INFO)
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.val_gen = val_gen
        self.steps_per_val = steps_per_val or len(val_gen)
        self.target_scaler = target_scaler or DummyScaler()
        self.best_epoch = 0

        if isinstance(monitor, Callable):
            self.metric = monitor
            self.monitor = monitor.__name__
        elif monitor == "mae":
            self.metric = mae
            self.monitor = "mae"
        else:
            raise ValueError(f'Unknown metric {monitor}')

        if log_monitor is not None:
            if isinstance(log_monitor, Callable):
                self.log_metric = log_monitor
                self.log_monitor = log_monitor.__name__
            else:
                raise TypeError()
        else:
            self.log_metric = mae
            self.log_monitor = 'mae'
        
        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch: int, logs: Dict = None) -> None:
        """
        Codes called by the callback at the end of epoch
        Args:
            epoch (int): epoch id
            logs (dict): logs of training
        Returns:
            None
        """
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            val_pred = []
            val_y = []
            for i in range(self.steps_per_val):
                val_data = self.val_gen[i]  # type: ignore
                nb_atom = _count(np.array(val_data[0][-2]))
                stop_training = self.model.stop_training  # save stop_trainings state
                pred_ = self.model.predict(val_data[0])
                self.model.stop_training = stop_training
                val_pred.append(self.target_scaler.inverse_transform(pred_[0, :, :], nb_atom[:, None]))
                val_y.append(self.target_scaler.inverse_transform(val_data[1][0, :, :], nb_atom[:, None]))
            current = self.metric(np.concatenate(val_y, axis=0), np.concatenate(val_pred, axis=0))
            if isinstance(current, tf.Tensor):
                current = current.numpy()
            if self.log_metric is not None:
                current_log = self.log_metric(np.concatenate(val_y, axis=0), np.concatenate(val_pred, axis=0))
                if isinstance(current_log, tf.Tensor):
                    current_log = current_log.numpy()
            else:
                current_log = None
            filepath = self.filepath.format(**{"epoch": epoch + 1, self.monitor: current})
            logger.info(f"val {self.monitor}: {current:.5f} {self.log_monitor}: {current_log:.5f}")

            if self.save_best_only:
                if current is None:
                    warnings.warn(f"Can save best model only with {self.monitor} available, skipping.", RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        logger.info(
                            f"best {self.monitor}: {self.best:.5f} -> {current:.5f}"
                        )
                        self.best = current
                        self.best_epoch = epoch
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        pass
            else:
                # logger.info(f"\nEpoch {epoch+1:05d}: saving model to {filepath}")
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
            
            logpath = filepath[:-5] + ".json"
            if Path(logpath).exists():
                log = loadfn(logpath)
            else:
                log = []
            log.append({
                'epoch': epoch+1, 
                f'val_{self.monitor}': current, 
                f'val_{self.log_monitor}': current_log,
                f'best_{self.monitor}': self.best,
                'best_epoch':  self.best_epoch+1
            })
            dumpfn(log, logpath)
                

class ManualStop(Callback):
    """
    Stop the training manually by putting a "STOP" file in the directory
    """

    def on_batch_end(self, epoch: int, logs: Dict = None) -> None:
        """
        Codes called at the end of a batch
        Args:
            epoch (int): epoch id
            logs (Dict): log dict
        Returns: None
        """
        if os.path.isfile("STOP"):
            self.model.stop_training = True


class ReduceLRUponNan(Callback):
    """
    This callback function solves a problem that when doing regression,
    an nan loss may occur, or the loss suddenly shoot up.
    If such things happen, the model will reduce the learning rate
    and load the last best model during the training process.
    It has an extra function that patience for early stopping.
    This will move to indepedent callback in the future.
    """

    def __init__(
        self,
        filepath: str = "./callback/val_mae_{epoch:05d}_{val_mae:.6f}.hdf5",
        factor: float = 0.5,
        verbose: bool = True,
        patience: int = 500,
        monitor: str = "val_mae",
        mode: str = "auto",
        has_sample_weights: bool = False,
    ):
        """
        Args:
            filepath (str): filepath for saved model checkpoint, should be consistent with
                checkpoint callback
            factor (float): a value < 1 for scaling the learning rate
            verbose (bool): whether to show the loading event
            patience (int): number of steps that the val mae does not change.
                It is a criteria for early stopping
            monitor (str): target metric to monitor
            mode (str): min, max or auto
            has_sample_weights (bool): whether the data has sample weights
        """
        self.filepath = filepath
        self.verbose = verbose
        self.factor = factor
        self.losses: deque = deque([], maxlen=10)
        self.patience = patience
        self.monitor = monitor
        super().__init__()

        if mode == "min":
            self.monitor_op = np.argmin
        elif mode == "max":
            self.monitor_op = np.argmax
        else:
            if "acc" in self.monitor:
                self.monitor_op = np.argmax
            else:
                self.monitor_op = np.argmin

        # get variable name
        variable_name_pattern = r"{(.+?)}"
        self.variable_names = re.findall(variable_name_pattern, filepath)
        self.variable_names = [i.split(":")[0] for i in self.variable_names]
        self.has_sample_weights = has_sample_weights
        if self.monitor not in self.variable_names:
            raise ValueError("The monitored metric should be in the name pattern")

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """
        Check the loss value at the end of an epoch
        Args:
            epoch (int): epoch id
            logs (dict): log history
        Returns: None
        """
        logs = logs or {}
        loss = logs.get("loss")
        last_saved_epoch, last_metric, last_file = self._get_checkpoints()
        if last_saved_epoch is not None:
            if last_saved_epoch + self.patience <= epoch:
                self.model.stop_training = True
                logger.info(f"{self.monitor} does not improve after {self.patience}, stopping the fitting...")

        if loss is not None:
            self.losses.append(loss)
            if np.isnan(loss) or np.isinf(loss):
                if self.verbose:
                    logger.info("Nan loss found!")
                self._reduce_lr_and_load(last_file)
                if self.verbose:
                    logger.info(f"Now lr is {float(kb.eval(self.model.optimizer.lr))}.")
            else:
                if len(self.losses) > 1:
                    if self.losses[-1] > (self.losses[-2] * 100):
                        self._reduce_lr_and_load(last_file)
                        if self.verbose:
                            logger.info(
                                f"Loss shot up from {self.losses[-2]:.3f} to {self.losses[-1]:.3f}! Reducing lr "
                            )
                            logger.info(f"Now lr is {float(kb.eval(self.model.optimizer.lr))}.")

    def _reduce_lr_and_load(self, last_file):
        old_value = float(kb.eval(self.model.optimizer.lr))
        self.model.reset_states()
        self.model.optimizer.lr = old_value * self.factor

        if last_file is not None:
            self.model.load_weights(last_file)
            if self.verbose:
                logger.info(f"Load weights {last_file}")
        else:
            logger.info("No weights were loaded")

        opt_dict = self.model.optimizer.get_config()
        sample_weight_model = "temporal" if self.has_sample_weights else None
        self.model.compile(
            self.model.optimizer.__class__(**opt_dict), self.model.loss, sample_weight_mode=sample_weight_model
        )

    def _get_checkpoints(self):
        file_pattern = re.sub(r"{(.+?)}", r"([0-9\.]+)", self.filepath)
        glob_pattern = re.sub(r"{(.+?)}", r"*", self.filepath)
        all_check_points = glob(glob_pattern)

        if len(all_check_points) > 0:
            metric_index = self.variable_names.index(self.monitor)
            epoch_index = self.variable_names.index("epoch")
            metric_values = []
            epochs = []
            for i in all_check_points:
                metrics = re.findall(file_pattern, i)[0]
                metric_values.append(float(metrics[metric_index]))
                epochs.append(int(metrics[epoch_index]))
            ind = self.monitor_op(metric_values)
            return epochs[ind], metric_values[ind], all_check_points[ind]
        return None, None, None


def _count(a: np.ndarray) -> np.ndarray:
    """
    count number of appearance for each element in a
    Args:
        a: (np.array)
    Returns:
        (np.array) number of appearance of each element in a
    """
    a = a.ravel()
    a = np.r_[a[0], a, np.Inf]
    z = np.where(np.abs(np.diff(a)) > 0)[0]
    z = np.r_[0, z]
    return np.diff(z)


class SGDRScheduler(tf.keras.callbacks.Callback):
    """Cosine annealing learning rate scheduler with periodic restarts.
    https://mancap314.github.io/cyclical-learning-rates-with-tensorflow-implementation.html

    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, 
                        epochs=100, 
                        callbacks=[schedule])
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range.
        max_lr: The upper bound of the learning rate range.
        lr_decay: Reduce the max_lr after 
                        completion of each cycle.
                  Ex. To reduce the max_lr by 20% 
                        after each cycle, set 
                        this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each 
                        full cycle completion.

    # References
        Original paper: http://arxiv.org/abs/1608.03983
    """
    def __init__(self,
                 min_lr,
                 max_lr,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart \
                / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 \
            * (self.max_lr - self.min_lr) \
            * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the 
        minimum value at the start of training.'''
        self.steps_per_epoch = self.params['steps'] \
            if self.params['steps'] is not None \
            else round(self.params['samples'] \
                        / self.params['batch_size'])
        logs = logs or {}
        tf.keras.backend.set_value(self.model.optimizer.lr, 
                                    self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics 
        and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(
            tf.keras.backend.get_value(
                self.model.optimizer.lr
            )
        )
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        tf.keras.backend.set_value(
            self.model.optimizer.lr, 
            self.clr()
        )

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, 
        apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(
                self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

    # def on_train_end(self, logs={}):
    #     '''Set weights to the values from the end of 
    #     the most recent cycle for best performance.'''
    #     self.model.set_weights(self.best_weights)
