import logging as log
import os
from itertools import cycle

import numpy as np
from keras.layers import TimeDistributed, Bidirectional, LSTM, Dense, Masking
from keras.models import Sequential

from experiments.data_utils import unpickle, split
from experiments.marking.sentence_encoder import SentenceEncoder
from experiments.sequencenet import SequenceNet
from experiments.tags import CategoricalTags


class ClassifierNet(SequenceNet):
    def compile_model(self):
        output_len = 150
        blstm_cell_size = 20
        lstm_cell_size = 20

        m = Sequential()
        m.add(Masking(input_shape=(self.timesteps, self.x_len)))
        m.add(TimeDistributed(Dense(output_len, activation='tanh'),
                              input_shape=(self.timesteps, self.x_len)))
        m.add(Bidirectional(LSTM(blstm_cell_size, return_sequences=True),
                            merge_mode='concat'))
        m.add(LSTM(lstm_cell_size, return_sequences=False))
        m.add(Dense(self.nbclasses, activation='softmax'))

        m.compile(loss='categorical_crossentropy', optimizer='adam', sample_weight_mode='temporal')
        self._model = m

    # todo: test predicting
    def predict_batch(self, tokenized_texts):
        texts_enc = [self.padder.pad(self._encoder.encode_data(ttext))[0] for ttext in tokenized_texts]
        x = np.array(texts_enc)
        batch_size = len(x)

        classes_batch = self._model.predict_classes(x, batch_size=batch_size)
        return classes_batch


if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

    raw_tags = (0, 1)
    tags = CategoricalTags(raw_tags)
    encoder = SentenceEncoder(tags)
    net = ClassifierNet(encoder, timesteps=150, batch_size=16)
    net.compile_model()

    data_file_path = net.relpath('data/data_tagged.pck')
    if os.path.isfile(data_file_path):
        data_thing = unpickle(data_file_path)
        data = list(data_thing)
        splits = (0.1, 0.2, 0.7)
        data_splits = split(data, splits)
        data_val = cycle(data_splits[0])
        data_test = cycle(data_splits[1])
        data_train = cycle(data_splits[2])

        net.train(data_train_gen=data_train, epoch_size=1024, epochs=10,
                  data_val_gen=data_val, nb_val_samples=128)
    else:
        log.warning('ClassifierNet training: file {} with data was not found! Please, create dataset first.')
