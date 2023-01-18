"""
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import typing
from abc import abstractmethod
from dataclasses import dataclass, field, InitVar
from time import time
from typing import Any, Dict, List, Optional, Type, Union, cast

from rt_core.utils import profiler
from rt_core.engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation

class Pipeline:
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class."""

    # pylint: disable=abstract-method

    def setup(self, model, optim):
        # to keep track of which device the nn.Module is on
        self._model = model
        self._optim = optim
        self._train_num_items_per_batch = 1
        """Expected number of items per batch used during the training
           Warning: Must be > 0
        """

    @property
    def model(self):
        return self._model

    @property
    def optim(self):
        return self._optim

    @property
    def train_num_items_per_batch(self):
        return self._train_num_items_per_batch

    @train_num_items_per_batch.setter
    def train_num_items_per_batch(self, value):
        self._train_num_items_per_batch = value

    @profiler.time_function
    def train(self, step: int, writer_use: InitVar):
        """Train the model

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        self.func_train(step, writer_use)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """ Example
        args
        training_callback_attributes: Attributes that can be passed between callbacks
        """
        callbacks = []
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.before_train,
            )
        )
        # add custom callbacks
        callbacks = self.get_custom_callbacks(callbacks, training_callback_attributes)
        return callbacks

    def get_custom_callbacks(
        self, callbacks, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        return callbacks

    def before_train(self, step: int):
        pass

    @abstractmethod
    def func_train(self, step:int, writer_use: InitVar):
        """
        Return: Nothing
        """
        pass
