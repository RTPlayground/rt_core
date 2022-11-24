# RLPlayground: ``rt_core``

``rt_core`` is the core Python library of RTPlayground that provides an OpenAI-like external API for interacting with real world and simulated robots. It also provides a centralized logging API for easy comparison between algorithms.

- [``rt_core`` abstract](https://github.com/RTPlayground/rt_core/blob/main/etc/abstract/abstract.pdf): High level design and objective.
- [scripts/](https://github.com/RTPlayground/rt_core/blob/main/scripts): Top-level scripts

## Usage

``rt_core`` can be used as an API as follows.
```python
import rt_core

class YourEnv(rt_core.Env): # inherit rt_core.Env
# Failure to define important functions will raise an error (or a warning) upon creating an instance of the class
    def ...

class YourTask(rt_core.Task): # inherit rt_core.Task
    def ...

interface = Interface(env=YourEnv(), task=YourTask()) 

# Use openAI gym-like functionalities
interface.reset()
interface.step()
interface.render()
interface.state # return state
interface.obs # return observations
interface.close()
interface.log() # centralized logger
```

## Install

```console
git clone https://github.com/RTPlayground/rt_core
cd rt_core
pip3 install .
```

## Uninstall

```console
pip3 uninstall rt_core
```

## Ignore Warnings

Warnings are meant to guide environment/task developers in fully implementing the environment (according to OpenAI Gym's Env standards), however if you would like to ignore the warnings you can add the following lines to your python script.

```python
import warnings
warnings.simplefilter('ignore')
```
