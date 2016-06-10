from noh.component import Component

class Binder(object):
    """ Binds a constant value to a keyword argument.

    Given a key-value pair on instantiation, binds the value as a constant value
    to a keyword argument for the bound :class: `Component`.

    """

    def __init__(self, **kwargs):
        self.args = kwargs

    def __call__(self, args):
        """ Assign values to argument Dictionary.

        Args:
            args (Dictionary): a Dictionary to copy the values to.

        Returns:
            A Dictionary containing the copied values.

        """

        for arg in self.args:
            args[arg] = self.args[arg]
        return args

class Mapper(object):
    """ Maps a keyword argument to another keyword.

    Given a key-value pair on instantiation, binds the given maps the values of
    the key keyword to the value keyword.

    """

    def __init__(self, **kwargs):
        self.args = kwargs

    def __call__(self, args, **kwargs):
        """ Assign values to argument Dictionary.

        Args:
            args (Dictionary): a Dictionary to copy the values to.

        Returns:
            A Dictionary containing the copied values.

        """

        for arg in self.args:
            if arg in kwargs:
                key = self.args[arg]
                args[key] = kwargs[arg]
                if key != arg:
                    del args[arg]
        return args

class Wrapper(Component):
    """ Wrap a component to call and train on bound values.

    Creates a new :class: `Component` that passes appropriate values as keyword
    arguments to the :meth: `__call__` and :meth: `train` of the base class.

    .. admonition:: Example

        This is a simple example of wrapping a :class: `Component`. This is
        especially useful when creating :class: `Circuit`s which include :class:
        `Component`s where each :class: `Component` requires different settings
        for training (e.g. learning rate, labeled data). Here we show an example
        of combining multiple autoencoders and connecting a perceptron layer::

            import numpy as np
            import noh
            import noh.components as C

            import data # From Chainer MNIST sample

            stacked_autoencoder = noh.Circuit(
                noh.Wrapper(C.Autoencoder(784, 1000), train=[noh.Binder(epochs=10)]),
                noh.Wrapper(C.Autoencoder(1000, 1000), train=[noh.Binder(epochs=10)]),
                noh.Wrapper(C.Perceptron(1000, 10), train=['labels', noh.Binder(epochs=10)]),
            )

            # Assume this function loads MNIST data
            mnist = data.load_mnist_data()
            mnist['data'] = mnist['data'].astype(np.float32)
            mnist['data'] /= 255
            mnist['target'] = mnist['target'].astype(np.int32)

            N_train = 60000
            x_train, x_test = np.split(mnist['data'],   [N_train])
            y_train, y_test = np.split(mnist['target'], [N_train])
            N_test = y_test.size

            errors = stacked_autoencoder.train(x_train, labels=y_train)

    Args:
        component: The Component to wrap.
        call: List of strings or Binder or Mapper

    """

    def __init__(self, component, call=[], train=[]):
        self.component = component
        self.call_args = call
        self.train_args = train

    def __call__(self, data, **kwargs):
        args = {}
        if len(self.call_args) == 0:
            return self.component(data)
        for arg in self.call_args:
            if isinstance(arg, Binder):
                args = arg(args)
            elif isinstance(arg, Mapper):
                args = arg(args, **kwargs)
            elif arg in kwargs:
                args[arg] = kwargs[arg]
        return self.component(data, **args)

    def train(self, data, **kwargs):
        args = {}
        if len(self.train_args) == 0:
            return self.component.train(data)
        for arg in self.train_args:
            if isinstance(arg, Binder):
                args = arg(args)
            elif isinstance(arg, Mapper):
                args = arg(args, **kwargs)
            elif arg in kwargs:
                args[arg] = kwargs[arg]
        return self.component.train(data, **args)
