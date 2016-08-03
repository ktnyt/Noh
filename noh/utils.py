import numpy as np

def labelize(components):
    labels = [component.name for component in components]
    if len(labels) != len(set(labels)):
        tmp = []
        counts = {}
        for label in labels:
            if labels.count(label) > 1:
                if label not in counts:
                    counts[label] = 0
                else:
                    counts[label] += 1
                tmp.append('{}{}'.format(label, counts[label]))
            else:
                tmp.append(label)
        labels = tmp
    return labels

class Collection(object):
    keys = []
    values = []

    def __init__(self, keys, values):
        if keys is None:
            keys = ['c{}'.format(i) for i in range(len(values))]

        assert(len(keys) == len(set(keys)))
        assert(len(keys) == len(values))

        self.keys = keys
        self.values = values

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
