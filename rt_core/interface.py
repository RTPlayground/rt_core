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
        return self.env.render()

    def close(self):
        self.env.close()

    def log(self, key, value):
        pass

    def inspect(self, who, what):
        '''
        Get the list of arguments
        https://stackoverflow.com/questions/582056/getting-list-of-parameter-names-inside-python-function
        '''
        if hasattr(self, who):
            obj = getattr(self, who)
        else:
            warnings.warn(f"[w] cannot find({who}).", RuntimeWarning)
        if hasattr(self, who):
            class_method = getattr(obj, what)
        else:
            warnings.warn(f"[w] cannot find({what}).", RuntimeWarning)
        return class_method.__code__.co_varnames
        
    def query(self, who, what, *argument):
        '''
        Execute the command 'what' from object 'who', which accept a set of arguments.
        i.e. query('env','log','mykey',123)
        '''
        if hasattr(self, who):
            obj = getattr(self, who)
        else:
            warnings.warn(f"[w] cannot find({who}).", RuntimeWarning)
        if hasattr(self, who):
            class_method = getattr(obj, what)
        else:
            warnings.warn(f"[w] cannot find({what}).", RuntimeWarning)
        result = class_method(*argument)
        return result
        
