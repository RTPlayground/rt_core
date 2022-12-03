from abc import ABCMeta, abstractmethod

class Env(metaclass=ABCMeta):
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self): #,action  # Preserved the interface format
        pass

    def close(self):
        raise NotImplementedError()

    def get_obs(self):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

