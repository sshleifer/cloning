import pandas as pd

from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam
import helper

LEARNING_RATE = .0001


def add_conv_block(model, nb_filter, nb_row, nb_col, subsample=(2 ,2)):
    '''Add a conv layer followed by a max pooling layer'''
    model.add(Convolution2D(nb_filter, nb_row, nb_col,
                            activation='relu', border_mode='same', subsample=subsample))
    model.add(MaxPooling2D(pool_size=(2 ,2), strides=(1 ,1)))


def add_dense_block(model, n_hidden):
    '''Dense block with relu activation'''
    model.add(Dense(n_hidden, activation='relu'))

class NvidiaNet():

    def __init__(self, checkpoint_path='t1-weights-improvement-{epoch:02d}-{val_loss:.2f}.h5'):
        model = Sequential()

        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))
        add_conv_block(model, 24, 5, 5)
        add_conv_block(model, 36, 5, 5)
        add_conv_block(model, 48, 5, 5)
        add_conv_block(model, 64, 3, 3, subsample=(1,1))
        add_conv_block(model, 64, 3, 3, subsample=(1,1))

        model.add(Flatten())
        add_dense_block(model, 1164)
        model.add(Dropout(.3))
        model.add(BatchNormalization())
        add_dense_block(model, 100)
        model.add(Dropout(.3))
        add_dense_block(model, 50)
        add_dense_block(model, 10)

        model.add(Dense(1))
        model.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
        self.model = model

        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        self.callbacks_list = [checkpoint]

    def create_generators(self, driving_log, path, frac=0.2):
        d1_valid = driving_log.sample(frac=frac, replace=False)
        d1_tr = driving_log.drop(d1_valid.index)
        # TODO: use train_test_split

        self.number_of_samples_per_epoch = d1_tr.shape[0] * 3
        self.number_of_validation_samples = d1_valid.shape[0] * 3

        # create two generators for training and validation
        self.train_gen = helper.generate_next_batch(data=d1_tr, path=path)
        self.validation_gen = helper.generate_next_batch(data=d1_valid, path=path)

    def fit_generator(self, nb_epoch, **kwargs):
        return self.model.fit_generator(
            self.train_gen,
            self.number_of_samples_per_epoch,
            nb_epoch,
            validation_data=self.validation_gen,
            callbacks=self.callbacks_list,
            nb_val_samples=self.number_of_validation_samples,
            verbose=1,
            **kwargs
        )

if __name__ == '__main__':
    data = pd.read_csv('t2_train/driving_log.csv')
    clone_net = NvidiaNet()
    clone_net.create_generators(data, 't2_train')
    clone_net.fit_generator(2)