from noh.environment import UnsupervisedEnvironment
from PIL import Image
import numpy as np
import time
import os

class WalkingNao32Env(UnsupervisedEnvironment):

    n_visible = 32 * 32 * 3
    n_dataset = 26
    n_test_dataset = 0
    
    print os.getcwd()
    dir_name = "../noh/environments/walking_nao_32/"
    file_name_list = [str(i+1)+".bmp" for i in xrange(n_dataset)]
    dataset = []
    for file_name in file_name_list:
        print dir_name+file_name
        img = Image.open(dir_name+file_name)
        dataset.append(np.asarray(img).flatten())
    
    dataset = np.array(dataset) / 255.
    test_dataset = None

    print dataset.shape

    def __init__(self, model):
        super(WalkingNao32Env, self).__init__(model)

    def train(self, epochs=None):
        start_time = time.time()
        super(WalkingNao32Env, self).train(epochs)
        print "training time : ", (time.time() - start_time), "sec"

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

