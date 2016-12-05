import logging as log
import os
from itertools import cycle
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from experiments.data_common import Padding, BatchMaker, split


class SequenceNet:
    def __init__(self, encoder, timesteps, batch_size):
        self.x_len = encoder.vector_length
        self.batch_size = batch_size
        self.nbclasses = encoder.tags.nbtags
        self.timesteps = timesteps
        self._encoder = encoder
        self._model = None

        self.padder = Padding(pad_to_length=timesteps)
        self.batcher = BatchMaker(self.batch_size)

    @staticmethod
    def relpath(*path):
        """Return absolute path from path relative to directory where this file is contained.
        It is assumed that there other necessary files in that directory
        (e.g. dir with models, dir with data, dir with logs)"""
        return os.path.join(os.path.dirname(__file__), *path)

    @classmethod
    def from_model_file(cls, encoder, batch_size, model_path=None):
        if not model_path:
            model_dir = cls.relpath('models')
            model_name = '{}_model_full_default.h5'.format(cls.__name__.lower())
            model_path = os.path.join(model_dir, model_name)

        model = load_model(model_path)

        _, timesteps, x_len = model.layers[0].input_shape
        if x_len != encoder.vector_length:
            raise ValueError('encoder.vector_length is not consistent with the input of '
                             'loaded model ({} != {})'.format(encoder.vector_length, x_len))
        _, timesteps, nbclasses = model.layers[-1].output_shape
        if nbclasses != encoder.tags.nbtags:
            raise ValueError('encoder.tags.nbtags is not consistent with the output of '
                             'loaded model ({} != {})'.format(encoder.tags.nbtags, nbclasses))

        net = cls(encoder, timesteps, batch_size)
        net._model = model
        log.info('{}: loaded model from path {}'.format(cls.__name__, model_path))
        return net

    def train(self, data_train_gen, epochs, epoch_size,
              data_val_gen, nb_val_samples,
              save_epoch_models=True, dir_for_models='models',
              log_for_tensorboard=True, dir_for_logs='logs',
              max_q_size=2):
        """Train model, maybe with checkpoints for every epoch and logging for tensorboard"""
        log.info('{}: Training...'.format(type(self).__name__))

        data_train = self._make_data_gen(data_train_gen)
        data_val = self._make_data_gen(data_val_gen)
        callbacks = []
        # ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2)

        if save_epoch_models:
            filepath = self.relpath(
                dir_for_models,
                '{}_model_full_epochsize{}'.format(type(self).__name__.lower(), epoch_size)
                + '_epoch{epoch:02d}_valloss{val_loss:.2f}.h5'
            )
            mcheck_cb = ModelCheckpoint(filepath, verbose=0, save_weights_only=False, mode='auto')
            callbacks.append(mcheck_cb)
        if log_for_tensorboard:
            tb_cb = TensorBoard(log_dir=dir_for_logs, histogram_freq=1, write_graph=True, write_images=False)
            callbacks.append(tb_cb)

        return self._model.fit_generator(data_train,
                                         samples_per_epoch=epoch_size, nb_epoch=epochs,
                                         max_q_size=max_q_size,
                                         validation_data=data_val, nb_val_samples=nb_val_samples,
                                         callbacks=callbacks
                                         )

    def evaluate(self, data_gen, nb_val_samples, max_q_size=2):
        log.info('{}: Evaluating...'.format(type(self).__name__))
        data_val = self._make_data_gen(data_gen)
        return self._model.evaluate_generator(data_val, max_q_size=max_q_size, val_samples=nb_val_samples)

    def _make_data_gen(self, data, do_infinite=True):
        # Cycling for keras
        if do_infinite:
            data = cycle(data)
        data = self.padder(self._encoder(data))
        return self.batcher.batch_transposed(data)

    def save_model(self, path=None):
        if not path:
            path = self.relpath('models', '{}_model_full.h5'.format(type(self).__name__.lower()))
        self._model.save(path)


def train(net, data, epoch_size, nbepochs, nb_val_samples, splits=(0.1, 0.2, 0.7)):
    """
    Train neural network and save history of training
    :param net: neural network
    :param data: data
    :param epoch_size: size of the epoch
    :param nbepochs: number of epochs
    :param nb_val_samples: number of samples to use for validation each epoch
    :param splits: tuple of (val_samples, test_samples, train_samples) as part of data param
    """
    data_splits = split(data, splits)
    data_val = data_splits[0]
    data_test = data_splits[1]
    data_train = data_splits[2]

    # Training
    hist = net.train(data_train, nbepochs, epoch_size, data_val, nb_val_samples)
    print('Hisory:', hist.history)

    # Saving history
    hist_path = net.relpath(
        'logs/history_{}_epochsize{}_epochs{}.json'.format(type(net).__name__.lower(), epoch_size, nbepochs))
    with open(hist_path, 'w') as f:
        import json
        json.dump(hist.history, f)

    # Evaluating
    print('Evaluation: {}'.format(net.evaluate(data_test)))


