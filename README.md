# RLPlayground: ``rt_core``

``rt_core`` is the core Python library of RTPlayground that provides an OpenAI-like external API for interacting with real world and simulated robots. It also provides a centralized logging API for easy comparison between algorithms.

- [``rt_core`` abstract](https://github.com/RTPlayground/rt_core/blob/main/etc/abstract/abstract.pdf): High level design and objective.
- [scripts/](https://github.com/RTPlayground/rt_core/blob/main/scripts): Top-level scripts

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
