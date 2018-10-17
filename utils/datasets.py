import scipy.io as sp
import os
import numpy as np
from keras.utils import to_categorical

class Datasets():
    def __init__(self, root_path):
        self.root_path = root_path

    def __load_data__(self):
        pass

    def __split_train_test__(self):
        pass

    def get_batch(self):
        pass


class VLSC(Datasets):
    def __init__(self, root_path, test_split=0.3):
        super(VLSC, self).__init__(root_path)
        self.data, self.label, self.source_name = self.__load_data__()
        self.test_split = test_split
        self.data_train, self.data_test, self.label_train, self.label_test = self.__split_train_test__()
        self.nDims = self.data_train[0].shape[1]
        print('name\ttrain_shape\ttest_shape\t')
        for idx, (name) in enumerate(self.source_name):
            print(name, '\t', self.data_train[idx].shape, '\t', self.data_test[idx].shape)
        self.mean = None
        self.std = None

    def set_mean_std(self, testSource):
        sourceId = self.source_name.index(testSource)
        trainSamples = np.delete(self.data_train, sourceId)
        trainSamples = np.concatenate(trainSamples, axis=0)
        self.mean = np.mean(trainSamples, axis=0)
        self.std = np.std(trainSamples, axis=0)


    def __load_data__(self):
        V = sp.loadmat(os.path.join(self.root_path, 'VOC2007.mat'))['data']
        L = sp.loadmat(os.path.join(self.root_path, 'LabelMe.mat'))['data']
        S = sp.loadmat(os.path.join(self.root_path, 'SUN09.mat'))['data']
        C = sp.loadmat(os.path.join(self.root_path, 'Caltech101.mat'))['data']
        data = np.array([V[:, :-1], L[:, :-1], S[:, :-1], C[:, :-1]])
        label = np.array([V[:, -1], L[:, -1], S[:, -1], C[:, -1]])
        label = [item - 1 for item in label]
        return data, label, ['V', 'L', 'S', 'C']

    def __split_train_test__(self):
        data_train = []
        data_test = []
        label_train = []
        label_test = []
        for i in range(len(self.data)):
            length = self.data[i].shape[0]
            test_size = int(self.test_split * length)
            test_idx = np.random.choice(np.arange(length), test_size, replace=False)
            data_test.append(self.data[i][test_idx])
            data_train.append(np.delete(self.data[i], test_idx, axis=0))
            label_test.append(self.label[i][test_idx])
            label_train.append(np.delete(self.label[i], test_idx, axis=0))
        return np.array(data_train), np.array(data_test), np.array(label_train), np.array(label_test)

    def generator(self, testSource, batch_size=32):
        sourceId = self.source_name.index(testSource)
        trainSamples = np.delete(self.data_train, sourceId)
        trainLabels = np.delete(self.label_train, sourceId)
        trainDomainIds = [np.ones(trainLabels[i].shape) * i for i in range(len(trainLabels))]

        trainSamples = [(item - self.mean)/self.std for item in trainSamples]
        while True:
            # sample one batch from each source domain
            sampleId = [np.random.choice(np.arange(len(item)), batch_size,replace=False) for item in trainLabels]
            batch_x = np.concatenate(
                [trainSamples[i][sampleId[i]] for i in range(len(sampleId))], axis=0
            )
            batch_y = np.concatenate(
                [trainLabels[i][sampleId[i]] for i in range(len(sampleId))], axis=0
            )
            batch_d = np.concatenate(
                [trainDomainIds[i][sampleId[i]] for i in range(len(sampleId))], axis=0
            )
            # print(np.unique(batch_d))
            yield (batch_x.astype(float), [batch_x.astype(float), batch_d.astype(int),
                                           np.ones((batch_y.shape[0], 1)),
                                           batch_y.astype(int)])

    def adversarialGenerator(self, testSource, model, batch_size):
        sourceId = self.source_name.index(testSource)
        trainSamples = np.concatenate(np.delete(self.data_train, sourceId), axis=0)
        trainSamples = (trainSamples - self.mean) / self.std
        _, trainSamples, _, _ = model.predict(trainSamples)
        tol_index = np.arange(trainSamples.shape[0])

        while True:
            sample_label = np.ones([batch_size, 1])
            source_label = np.zeros([batch_size, 1])
            source_idx = np.random.choice(tol_index, size=batch_size, replace=False)
            source_data = trainSamples[source_idx]
            sample_data = np.random.laplace(0, 1, size=source_data.shape)
            yield np.concatenate((source_data, sample_data), axis=0).astype(float), \
                  np.concatenate((source_label, sample_label), axis=0).astype(int)

    def getValData(self, testSource):
        sourceId = self.source_name.index(testSource)
        valSamples = np.delete(self.data_test, sourceId)
        valLabels = np.delete(self.label_test, sourceId)
        valDomainIds = [np.ones(valLabels[i].shape) * i for i in range(len(valLabels))]
        valSamples = [(item - self.mean)/self.std for item in valSamples]
        s, l, d = np.concatenate(valSamples, axis=0), \
               np.concatenate(valLabels, axis=0).astype(int), \
               np.concatenate(valDomainIds, axis=0).astype(int)
        index = np.arange(s.shape[0])
        np.random.shuffle(index)
        s = s[index]
        l = l[index]
        d = d[index]
        return s, l, d

    def getTestData(self, testSource):
        sourceId = self.source_name.index(testSource)
        testSamples = self.data_test[sourceId]
        testLabels = self.label_test[sourceId]
        testSamples = (testSamples - self.mean) / self.std
        return testSamples, testLabels.astype(int)
