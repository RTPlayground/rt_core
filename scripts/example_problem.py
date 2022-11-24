from rt_core import Interface, Env, Task
import random

class ThreeBlock(Env):
    def __init__(self):
        self.reset()
    
    def step(self, action):
        if(action > 0 and self.state[2] != 1):
            self.state = [0] + self.state[:2]
        elif(action < 0 and self.state[0] != 1):
            self.state = self.state[1:] + [0]
        else:
            pass
        return self.state

    def reset(self):
        self.state = [1,0,0]
        return self.state

class GoRight(Task):
    def __init__(self):
        pass

    def get_done(self, state):
        if(state[2] == 1):
            return True
        else:
            return False

    def get_reward(self, state):
        if(state[2] == 1):
            return 1
        else:
            return -0.1

int = Interface(env = ThreeBlock(), task = GoRight())

state = int.reset()
done = False
print("==================")
while not done:
    action = random.randint(-1,1)
    state, reward, done = int.step(action)
    print("action:", action)
    print("state:", state)
    print("reward:", reward)
    print("done:", done)
    print("==================")
