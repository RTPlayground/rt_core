#!/usr/bin/env python
import io
import os
import re
from datetime import datetime
from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    """Adapted from https://github.com/amazon-science/earth-forecasting-transformer/blob/main/setup.py"""
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('rt_core', '__init__.py')
if VERSION.endswith('dev'):
    VERSION = VERSION + datetime.today().strftime('%Y%m%d')
    
# Read requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    # Metadata
    name='rt_core',
    version=VERSION,
    python_requires='>=3.7',
    description='rt_core is the core Python library of RTPlayground that provides an OpenAI-like external API for interacting with real world and simulated robots. It also provides a centralized logging API for easy comparison between algorithms.',
    license='MIT',
    zip_safe=True,
    include_package_data=True,
    packages=['rt_core'],
    scripts=[],
    url="https://github.com/RTPlayground/rt_core/",
    install_requires=requirements,
    # https://gist.github.com/nazrulworld/3800c84e28dc464b2b30cec8bc1287fc
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    author='Rom Parnichkun, Alessandro Moro',
    author_email='rom.parnichkun@gmail.com, alessandromoro.italy@gmail.com',
)
