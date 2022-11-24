from abc import ABCMeta, abstractmethod

class Env(metaclass=ABCMeta):
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def get_obs(self):
        pass

