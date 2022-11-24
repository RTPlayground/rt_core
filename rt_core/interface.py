import wandb
import warnings

class Interface:
    def __init__(self, env, task):
        self.env = env
        self.task = task

        if hasattr(env, "action_space"):
            self.action_space = self.env.action_space
        else:
            warnings.warn(f"action_space not declared in the Environment Module ({env}).", RuntimeWarning)
        if hasattr(env, "observation_space"):
            self.observation_space = self.env.observation_space
        else:
            warnings.warn(f"observation_space not declared in the Environment Module ({env}).", RuntimeWarning)
        if hasattr(env, "state"):
            pass
        else:
            warnings.warn(f"state variable not declared in the Environment Module ({env}).", RuntimeWarning)
        if hasattr(task, "reward_range"):
            self.reward_range = self.task.reward_range
        else:
            warnings.warn(f"reward_range not declared in the Task Module ({task}).", RuntimeWarning)

    @property
    def state(self):
        return self.env.state

    @property
    def obs(self):
        return self.env.get_obs()

    def step(self, action):
        s_next = self.env.step(action)
        reward = self.task.get_reward(s_next)
        done = self.task.get_done(s_next)
        return s_next, reward, done

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def log(self, key, value):
        pass
