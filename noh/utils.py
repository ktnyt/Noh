import numpy as np

class DotAccessible(object):
    def __init__(self, obj):
        self.obj=obj

    def __getitem__(self, i):
        return self.wrap(self.obj[i])

    def __getslice__(self, i, j):
        return map(self.wrap, self.obj.__getslice__(i,j))

    def __getattr__(self, key):
        if isinstance(self.obj, dict):
            try:
                v=self.obj[key]
            except KeyError:
                v=self.obj.__getattribute__(key)
        else:
            v=self.obj.__getattribute__(key)

        return self.wrap(v)

    def wrap(self, v):
        if isinstance(v, (dict,list,tuple)): # xx add set
            return self.__class__(v)
        return v

def get_lr_func(lr_type="const", lr=None, r_div=None):

    if lr_type == "const":
        if lr is None:
            raise ValueError("lr should be decided in case lr_type is \"const\" ")
        def opt_const(**kwargs):
            return lr
        return opt_const

    elif lr_type == "hinton_r_div":
        if r_div is None:
            raise ValueError("r_div should be decided in case lr_type is \"hinton_r_div\" ")
        #def opt_hinton(weight, d_weight):
        def opt_hinton(**kwargs):
            # print "abs weight   : ", np.sum(np.abs(kwargs["weight"]))
            # print "abs d_weight : ", np.sum(np.abs(kwargs["d_weight"]))
            return np.sum(np.abs(kwargs["weight"])) / (np.sum(np.abs(kwargs["d_weight"])) * r_div)
        return opt_hinton
    else:
        raise ValueError("{0} is not defined.".format(lr_type))


def get_standardized_rewards(rewards):
    rewards -= np.mean(rewards)
    rewards /= np.std(rewards)
    return rewards


def get_discounted_rewards(rewards, gamma, reward_reset_checker=None):

    if reward_reset_checker is None:
        reward_reset_checker = lambda x: False

    discounted_reward_list = np.zeros_like(rewards)
    discounted_reward = 0.
    for t in reversed(xrange(0, rewards.size)):
        if reward_reset_checker(rewards):
            discounted_reward = 0.
        discounted_reward = (discounted_reward * gamma) + rewards[t]
        discounted_reward_list[t] = discounted_reward

    return discounted_reward_list