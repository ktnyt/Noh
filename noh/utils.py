import numpy as np

class Collection(object):
    keys = []
    values = []

    def __init__(self, collection):
        if isinstance(collection, dict):
            for key, value in collection.items():
                self.keys.append(key)
                self.values.append(key)

        if isinstance(collection, list):
            keys = [item.__class__.__name__.lower() for item in collection]

            counts = {}

            for key, value in zip(keys, collection):
                if keys.count(key) > 1:
                    if key not in counts:
                        counts[key] = 0
                    counts[key] += 1
                    key = '{}{}'.format(key, counts[key])
                self.keys.append(key)
                self.values.append(value)

    def __getitem__(self, key):
        if isinstance(key, int):
            index = key
        else:
            index = self.keys.index(key)
        return self.values[index]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            index = key
        else:
            index = self.keys.index(key)
        self.values[index] = value

    def __delitem__(self, key):
        if isinstance(key, int):
            index = key
        else:
            index = self.keys.index(key)
        del self.keys[key]
        del self.values[key]

    def __iter__(self):
        return iter(self.keys)

    def __getslice__(self, i, j):
        raise NotImplementedError("To be implemented")

    def __setslice__(self, i, j, values):
        raise NotImplementedError("To be implemented")

    def __delslice__(self, i, j):
        raise NotImplementedError("To be implemented")

    def __getattr__(self, key):
        return self.__getitem__(key)

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
