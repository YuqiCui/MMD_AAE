from .losses import MMD_Loss_func, adjust_binary_cross_entropy
from keras.losses import binary_crossentropy, mean_squared_error
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam


class Basic_MMD_AAE():
    def __init__(self, nSources, sigmas):
        self.loss = MMD_Loss_func(nSources, sigmas)

    def encoder(self):
        pass

    def decoder(self):
        pass

    def taskOut(self):
        pass

    def adversarial(self):
        pass

    def makeModel(self):
        pass


class MMD_AAE(Basic_MMD_AAE):
    def __init__(self, nSources, input_shape, nClass,
                 sigmas=None, mmd=1, adv=1, ae=1,
                 cls=1, taskActivation='softmax',
                 taskLoss='sparse_categorical_crossentropy'):
        super(MMD_AAE, self).__init__(nSources, sigmas)
        self.input_shape = input_shape
        self.nClass = nClass
        self.weight_mmd = mmd
        self.weight_adv = adv
        self.weight_ae = ae
        self.weight_cls = cls
        self.hidden_layer = 2000

        # task
        self.taskActivation = taskActivation
        self.taskLoss = taskLoss

        self.E = self.encoder()
        self.D = self.decoder()(self.E.output)
        self.T = self.taskOut()(self.E.output)
        self.A = self.adversarial()
        self.Adv = self.A(self.E.output)
        self.A.compile('Adam', loss='binary_crossentropy', metrics=['acc'])

        model = Model(self.E.input, [self.D, self.E.output, self.Adv, self.T])
        # model.summary()
        model.compile(Adam(lr=10e-5),
                      loss={
                          'encoder': self.loss,
                          'decoder': mean_squared_error,
                          'adv': adjust_binary_cross_entropy,
                          'task': self.taskLoss
                      },
                      loss_weights={
                          'encoder': self.weight_mmd,
                          'decoder': self.weight_ae,
                          'adv': self.weight_adv,
                          'task': self.weight_cls
                      })
        # print(model.metrics_names)
        model = self.lockModel(model)
        self.model = model

    def encoder(self):
        model = Sequential()
        model.add(Dense(self.hidden_layer, input_shape=[self.input_shape, ], activation='linear', name='encoder'))
        return model

    def decoder(self):
        model = Sequential(name='decoder')
        model.add(Dropout(0.25, input_shape=[self.hidden_layer]))
        model.add(Dense(self.input_shape, input_shape=[self.hidden_layer], activation='linear'))
        return model

    def taskOut(self):
        model = Sequential(name='task')
        model.add(Dropout(0.25, input_shape=[self.hidden_layer, ]))
        model.add(Dense(self.hidden_layer, input_dim=self.hidden_layer, activation='relu'))
        model.add(Dense(self.nClass, activation=self.taskActivation))
        return model

    def adversarial(self):
        model = Sequential(name='adv')
        model.add(Dense(self.hidden_layer, input_dim=self.hidden_layer, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def makeModel(self):
        return self.model, self.A

    def lockModel(self, model):
        for layers in model.layers:
            if 'adv' not in layers.name:
                layers.trainable = True
            else:
                layers.trainable = False
        return model


if __name__ == "__main__":
    model = MMD_AAE(3, 4096, 5)

