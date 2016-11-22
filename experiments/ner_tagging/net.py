import logging as log
import numpy as np
from itertools import cycle, islice
from keras.models import Sequential, load_model
from keras.layers import TimeDistributed, Bidirectional, LSTM, Dense, Activation, Masking
from experiments.data_common import *
from experiments.ner_tagging.data_prep import data_gen


class NERNet:
    def __init__(self, x_len, timesteps=None, batch_size=1, nbclasses=3, path_to_model=None):
        self.x_len = x_len
        self.batch_size = batch_size
        self.nbclasses = nbclasses
        self.timesteps = timesteps
        self.model = None

        data_thing, encoder = data_gen(vector_length=self.x_len)
        self.encoder = encoder
        self._data_thing = data_thing
        self.padder = Padding(pad_to_length=self.timesteps)
        self.batch = BatchMaker(self.batch_size)

        self.data()
        if path_to_model:
            self.load_model(path_to_model)
        else:
            self.model()

    def data(self, splits=(0.1, 0.2, 0.7)):
        data_length = len(self._data_thing)
        edges = split_range(data_length, self.batch_size, splits)
        self.data_splits = []

        for a, b in edges:
            data_split = self.encoder(islice(self._data_thing.objects(), a, b))
            if self.timesteps:
                data_split = self.padder(data_split)
            data_split = cycle(self.batch(data_split))
            self.data_splits.append(data_split)

        self.data_val = self.data_splits[0]
        self.data_test = self.data_splits[1]
        self.data_train = self.data_splits[2]

    # todo: temp test method; remove then
    def data_test(self):
        data_thing, encoder = data_gen(vector_length=self.x_len)
        data_train = encoder(data_thing._wikiner_wp3())
        data_test = encoder(data_thing._wikigold_conll())
        if self.timesteps:
            pad = Padding(self.timesteps)
            data_train = pad(data_train)
            data_test = pad(data_test)
        batch = BatchMaker(self.batch_size)
        self.data_train = cycle(batch(data_train))
        self.data_val = cycle(batch(data_test))

    def model(self):
        output_len = 150
        blstm_cell_size = 20
        lstm_cell_size = 20

        m = Sequential()
        m.add(Masking(input_shape=(self.timesteps, self.x_len)))
        m.add(TimeDistributed(Dense(output_len, activation='tanh'),
                              input_shape=(self.timesteps, self.x_len)))
        m.add(Bidirectional(LSTM(blstm_cell_size, return_sequences=True),
                            merge_mode='concat'))
        m.add(LSTM(lstm_cell_size, return_sequences=True))
        m.add(TimeDistributed(Dense(self.nbclasses, activation='softmax')))

        m.compile(loss='categorical_crossentropy', optimizer='adam', sample_weight_mode='temporal')
        self.model = m

    def train(self, epochs, epoch_size=16384, nb_val_samples=256):
        log.info('NERNet: Training...')
        for ii, i in enumerate(range(0, epochs * epoch_size, epoch_size), 1):
            print('EPOCH {}/{}'.format(ii, epochs), '[{}, {})'.format(i, i+epoch_size))
            self.model.fit_generator(islice(self.data_train, 0, epoch_size),
                                     samples_per_epoch=epoch_size, nb_epoch=1,
                                     # max_q_size=2,
                                     validation_data=self.data_val, nb_val_samples=nb_val_samples,
                                     )

    # todo: maybe save state of the data generators (self.data_splits)
    def save_model(self, path='model_full.h5'):
        self.model.save(path)

    def load_model(self, path='model_full.h5'):
        log.info('NERNet: Loading model...')
        del self.model
        self.model = load_model(path)
        log.info('NERNet: Loaded model.')

    def evaluate(self, nb_val_samples=2048):
        log.info('NERNet: Evaluating...')
        self.model.evaluate_generator(self.data_val, val_samples=nb_val_samples)

    # todo: make more effective and for batchs
    def predict_single(self, tokenized_text):
        log.info('NERNet: Predicting {}'.format(tokenized_text))

        orig_len = len(tokenized_text)
        text_encoded = self.encoder.encode_text(tokenized_text)
        text_encoded = self.padder.pad(text_encoded)
        a = np.array(text_encoded[:1])
        predicted_all = self.model.predict(a, batch_size=1)

        predicted = predicted_all[0][:orig_len]
        log.debug('NERNet: Probabilites:\n', predicted)
        res = np.zeros_like(predicted)
        # replacing probabilities with category values
        res[range(len(predicted)), predicted.argmax(1)] = 1

        return self.encoder.decode_tags(res.tolist())


if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
    model_path = 'model_full.h5'
    net = NERNet(x_len=300, batch_size=32, nbclasses=3, timesteps=100, path_to_model=model_path)
    # net.evaluate(nb_val_samples=256)

    text = '010 is the tenth album from Japanese Punk Techno band The Mad Capsule Markets .'.split()
    text = np.array(text)
    truth = 'B 0 0 0 0 0 B 0 0 0 B I I I 0'.split()
    predicted = net.predict_single(text)
    print('text', text)
    print('pred', predicted)
    print('true', truth)

    # net.train(epochs=5)
    # net.save_model()
    # print(net.evaluate())

