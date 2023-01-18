# Copyright 2022 The Nerfstudio Team. All rights reserved.
# Copyright 2023 RTPlayground Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to train model.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
from rich.console import Console
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal


from rt_core.configs.experiment_config import ExperimentConfig
from rt_core.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
#from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from rt_core.utils import profiler, writer
from rt_core.utils.decorators import (
    check_eval_enabled,
    check_main_thread,
    check_viewer_enabled,
)
from rt_core.utils.misc import step_check
from rt_core.utils.writer import EventName, TimeWriter

from rt_core.pipelines.base_pipeline import Pipeline



CONSOLE = Console(width=120)

@dataclass
class TrainerConfig(ExperimentConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: Trainer)
    """target class to instantiate"""
    entity: str = 'i.e. wandb_entity_name'
    """Name assigned to the entity of the project (main target)"""
    project: str = 'project_noname'
    """Name assigned to this project (used in wandb)"""
    steps_per_save: int = 1000
    """Number of steps between saves."""
    steps_per_eval_batch: int = 500
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 500
    """Number of steps between single eval images."""
    steps_per_eval_all_images: int = 25000
    """Number of steps between eval all images."""
    max_num_iterations: int = 1000000
    """Maximum number of iterations to run."""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    save_only_latest_checkpoint: bool = True
    """Whether to only save the latest checkpoint or all checkpoints."""
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    """Optionally specify a pre-trained model directory to load from."""
    load_step: Optional[int] = None
    """Optionally specify model step to load from; if none, will find most recent model in load_dir."""
    load_config: Optional[Path] = None

class Trainer:
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
    """

    pipeline: Pipeline
    callbacks: List[TrainingCallback]

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.entity = config.entity
        self.project_name = config.project
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = "cpu" if world_size == 0 else f"cuda:{local_rank}"
        self.mixed_precision = self.config.mixed_precision
        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.mixed_precision)

        self.base_dir = config.get_base_dir()
        # directory to save checkpoints
        self.checkpoint_dir = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")
        # set up writers/profilers if enabled
        banner_messages = ['banner_message']
        writer_log_path = self.base_dir / config.logging.relative_log_dir
        writer.setup_event_writer(config.is_wandb_enabled(), config.is_tensorboard_enabled(), 
            entity=self.entity, projectname=self.project_name, log_dir=writer_log_path)
        writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations, banner_messages=banner_messages)
        # Write in the configuration the current given configuration class
        #writer.put_config(name="config", config_dict=dataclasses.asdict(config), step=0)
        profiler.setup_profiler(config.logging)

    def setup(self, pipeline, config):#, test_mode: Literal["test", "val", "inference"] = "val"):
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datset into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = pipeline
        if type(config) is dict:
            writer.put_config(name="config", config_dict=config, step=0)
        else:
            writer.put_config(name="config", config_dict=dataclasses.asdict(config), step=0)


        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                pipeline=self.pipeline,  # type: ignore
                writer_used=writer
            )
        )

    def train(self) -> None:
        """Train the model."""

        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations
            step = 0
            for step in range(self._start_step, self._start_step + num_iterations):
                with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:

                    # training callbacks before the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(
                            step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                        )

                    # time the forward pass
                    #loss, loss_dict = self.train_iteration(step)
                    self.train_iteration(step)

                    # training callbacks after the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:

                    writer.put_time(
                        name=EventName.TRAIN_ITEMS_PER_SEC,
                        duration=self.pipeline.train_num_items_per_batch / train_t.duration,
                        step=step,
                        avg_over_steps=True,
                    )

                #self._update_viewer_state(step)

                # a batch of train rays
                #if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                #    #writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                #    #writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                #    if step == 0 and self.pipeline.model is not None:
                #        # training callbacks after the training iteration
                #        for callback in self.callbacks:
                #            callback.run_callback_at_location(step, location=TrainingCallbackLocation.MODEL_SAVE_ITERATION)
                #            #writer.put_model(name='save model', model=self.pipeline.model, data_in=self.pipeline.last_input)

                # evaluate (not implemented yet)
                self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()
            # save checkpoint at the end of training
            self.save_checkpoint(step)

            CONSOLE.rule()
            CONSOLE.print("[bold green]:tada: :tada: :tada: Training Finished :tada: :tada: :tada:", justify="center")
            #if not self.config.viewer.quit_on_train_completion:
            #    CONSOLE.print("Use ctrl+c to quit", justify="center")
            #    #self._always_render(step)

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        CONSOLE.print("No checkpoints to load, training from scratch")

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        CONSOLE.print("Trainer::save_checkpoint: not implemented")

    @profiler.time_function
    def train_iteration(self, step: int) -> Dict[str, torch.Tensor]:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """
        # training callbacks after the training iteration
        #for callback in self.callbacks:
        #    callback.run_callback_at_location(step, location=TrainingCallbackLocation.MODEL_TEST_ITERATION)
        # training callbacks after the training iteration
        #for callback in self.callbacks:
        #    callback.run_callback_at_location(step, location=TrainingCallbackLocation.MODEL_TRAIN_ITERATION)

        loss = self.pipeline.train(step, writer)

        # Merging loss and metrics dict into a single output.
        #return loss, {'loss':loss}

    @profiler.time_function
    def eval_iteration(self, step: int) -> Dict[str, torch.Tensor]:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """
        pass
