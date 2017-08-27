import pandas as pd

from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam
import helper

LEARNING_RATE = .0001


def add_conv_block(model, nb_filter, nb_row, nb_col, subsample=(2 ,2)):
    model.add(Convolution2D(nb_filter, nb_row, nb_col,
                            activation='relu', border_mode='same', subsample=subsample))
    model.add(MaxPooling2D(pool_size=(2 ,2), strides=(1 ,1)))


def add_dense_block(model, n_hidden):
    model.add(Dense(n_hidden, activation='relu'))

class CloneNet():

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
        model.add(BatchNormalization())
        add_dense_block(model, 100)
        add_dense_block(model, 50)
        add_dense_block(model, 10)

        model.add(Dense(1))
        model.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
        self.model = model

        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        self.callbacks_list = [checkpoint]

    def create_generators(self, path='data/driving_log.csv'):
        d1 = pd.read_csv(path)
        d1_valid = d1.sample(frac=.20, replace=False)

        d1_tr = d1.drop(d1_valid.index)

        self.number_of_samples_per_epoch = d1_tr.shape[0] * 3
        self.number_of_validation_samples = d1_valid.shape[0] * 3

        # create two generators for training and validation
        self.train_gen = helper.generate_next_batch(data=d1_tr)
        self.validation_gen = helper.generate_next_batch(data=d1_valid)

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