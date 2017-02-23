import logging as log
import os
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from experiments.data_common import Padding, BatchMaker


class SequenceNet:
    def __init__(self, encoder, timesteps, batch_size):
        self.x_len = encoder.vector_length
        self.nbclasses = encoder.nbclasses
        self.timesteps = timesteps
        self.batch_size = batch_size
        self._encoder = encoder
        self._model = None

        self.padder = Padding(pad_to_length=timesteps)
        self.batcher = BatchMaker(self.batch_size)

    @staticmethod
    def relpath(*path):
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
              log_for_tensorboard=True, save_history=True, dir_for_logs='logs',
              max_q_size=2):
        """Train model, maybe with checkpoints for every epoch and logging for tensorboard"""
        log.info('{}: Training...'.format(type(self).__name__))

        data_train = self._make_data_gen(data_train_gen)
        data_val = self._make_data_gen(data_val_gen)
        callbacks = []

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

        hist = self._model.fit_generator(data_train,
                                         samples_per_epoch=epoch_size, nb_epoch=epochs,
                                         max_q_size=max_q_size,
                                         validation_data=data_val, nb_val_samples=nb_val_samples,
                                         callbacks=callbacks
                                         )
        if save_history:
            hist_path = self.relpath(
                'logs', 'history_{}_epochsize{}_epochs{}.json'.format(type(self).__name__.lower(), epoch_size, epochs))
            with open(hist_path, 'w') as f:
                import json
                json.dump(hist.history, f)

    def evaluate(self, data_gen, nb_val_samples, max_q_size=2):
        log.info('{}: Evaluating...'.format(type(self).__name__))
        data_val = self._make_data_gen(data_gen)
        return self._model.evaluate_generator(data_val, max_q_size=max_q_size, val_samples=nb_val_samples)

    def _make_data_gen(self, data):
        data = self.padder(self._encoder(data))
        return self.batcher.batch_transposed(data)

    def save_model(self, path=None):
        if not path:
            path = self.relpath('models', '{}_model_full.h5'.format(type(self).__name__.lower()))
        self._model.save(path)

