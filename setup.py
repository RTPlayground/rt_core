from setuptools import setup

setup(
    name='rt_core',
    version='0.0.1',
    author='Rom Parnichkun',
    author_email='rom.parnichkun@gmail.com',
    packages=['rt_core'],
    scripts=[],
    description='rt_core is the core Python library of RTPlayground that provides an OpenAI-like external API for interacting with real world and simulated robots. It also provides a centralized logging API for easy comparison between algorithms.',
    install_requires=[
       "wandb",
    ],
)
