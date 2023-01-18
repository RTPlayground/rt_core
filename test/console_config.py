from __future__ import annotations

#import sys
#sys.path.append('./')
#sys.path.append('../')

import yaml
import random
import socket
import traceback
from datetime import timedelta
from typing import Any, Callable, Optional
import dataclasses
import functools
import os
import time
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console

from rt_core.configs.config_utils import convert_markup_to_ansi
from rt_core.configs.method_configs import method_configs
from rt_core.utils import comms, profiler

from rt_core.engine.trainer import TrainerConfig
from rt_core.utils import writer
from rt_core.utils.writer import TimeWriter, EventName

from rt_core.utils.misc import step_check

CONSOLE = Console(width=120)
DEFAULT_TIMEOUT = timedelta(minutes=30)

def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(config: TrainerConfig) -> None:
    """Main function."""

    config.set_timestamp()
    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.dataparser.data")
        config.pipeline.datamanager.dataparser.data = config.data

    if config.load_config:
        CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
        config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)

    # print and save config
    config.print_to_terminal()
    config.save_config()

    # set up writers/profilers if enabled
    print('config.get_base_dir():{}'.format(config.get_base_dir()))
    print('config.logging.relative_log_dir:{}'.format(config.logging.relative_log_dir))
    writer_log_path = config.get_base_dir()# + '/log'# + config.logging.relative_log_dir
    writer.setup_event_writer(config.is_wandb_enabled(), config.is_tensorboard_enabled(), entity='utokyo-dlf', projectname = 'testproject', log_dir=writer_log_path)
    writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations, banner_messages=['banner_messages'])
    writer.put_config(name="config", config_dict=dataclasses.asdict(config), step=0)
    profiler.setup_profiler(config.logging)


    with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
        num_iterations = config.max_num_iterations
        step = 0
        _start_step = 0
        for step in range(_start_step,_start_step + num_iterations):
            with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                # train
                #time.sleep(1/1000)
                # training callbacks before the training iteration
                # time the forward pass
                # training callbacks after the training iteration
                pass
            # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
            if step > 1:
                train_num_rays_per_batch = 10 # debug number for visualization
                tmp_duration = train_t.duration
                if tmp_duration == 0: tmp_duration = 0.00001
                writer.put_time(
                    name=EventName.TRAIN_ITEMS_PER_SEC,
                    duration=train_num_rays_per_batch / tmp_duration,#train_t.duration,
                    step=step,
                    avg_over_steps=True,
                )
            # update_viewer_state
            # a batch of train rays
            steps_per_log = 10
            loss = 1.2
            lossloss_dict = {'loss_dict':2.3}
            metrics_dict = {'metrics_dict':3.4}
            if step_check(step, steps_per_log, run_at_zero=True):
                writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                writer.put_dict(name="Train Loss Dict", scalar_dict=lossloss_dict, step=step)
                writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)

            # eval
            if step_check(step, config.steps_per_save):
                #save_checkpoint(step)
                pass

            # write out the 
            writer.write_out_storage()
        # save checkpoint at the end of training
        #save_checkpoint(step)

        CONSOLE.rule()
        CONSOLE.print("[bold green]:tada: :tada: :tada: Training Finished :tada: :tada: :tada:", justify="center")
        if not config.viewer.quit_on_train_completion:
            CONSOLE.print("Use ctrl+c to quit", justify="center")



def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    config = method_configs['basic']
    main(config)


if __name__ == "__main__":
    entrypoint()
