from noh.environment import SupervisedEnvironment

import numpy as np
import time
import pylab
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report


class MNISTEnv(SupervisedEnvironment):

    n_visible = 28 * 28
    n_dataset = 60000
    n_test_dataset = 10000
    
    mnist = fetch_mldata('MNIST original', data_home="./mnist")
    X = mnist.data
    y = mnist.target
    
    X = X.astype(np.float64)
    X /= X.max()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.142857)
    
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)
    
    dataset = (X_train, labels_train)
    test_dataset = (X_test, labels_test)

    def __init__(self, model):
        super(MNISTEnv, self).__init__(model)

    def train(self, epochs=None):
        super(MNISTEnv, self).train(epochs)

    def unsupervised_train(self, epochs):
        self.model.unsupervised_trai(self.dataset, epochs)

    def show_images(self, dataset, labels=None):

        n_dataset = len(dataset)
        sqrt_n = np.ceil(np.sqrt(n_dataset))

        if labels is None:
            labels = ["" for i in xrange(n_dataset)]
        elif len(labels) is not n_dataset:
            raise ValueError("len(labels) should be equal to len(dataset).")

        for index, (data, label) in enumerate(zip(dataset, labels)):
            pylab.subplot(sqrt_n, sqrt_n, index+1)
            pylab.axis('off')
            pylab.imshow(data.reshape(28, 28), cmap=pylab.cm.gray_r, interpolation='nearest')
            pylab.title(label)

        pylab.show()
        
    def show_test_images(self, n_dataset=25):

        dataset = []
        labels = []
        for index, id in enumerate(np.random.random_integers(0, self.n_test_dataset, n_dataset)):
            dataset.append(self.test_dataset[0][id])
            labels.append("%i" % np.argmax(self.test_dataset[1][id]))

        self.show_images(np.array(dataset), np.array(labels))
            
    def show_reconstruct_images(self, n_dataset=18):

        dataset = []
        labels = []
        for index, id in enumerate(np.random.random_integers(0, self.n_test_dataset, n_dataset)):

            data = self.test_dataset[0][id]
            dataset.append(data)
            dataset.append(self.model.rec(data))

            label = self.test_dataset[1][id]
            labels.append('%i (origin)' % np.argmax(label))
            labels.append('%i (rec)' % np.argmax(label))

        self.show_images(np.array(dataset), np.array(labels))

