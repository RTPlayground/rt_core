from abc import ABCMeta, abstractmethod

class Task(metaclass=ABCMeta):
    @abstractmethod
    def get_reward(self, state):
        pass

    @abstractmethod
    def get_done(self, state):
        pass

