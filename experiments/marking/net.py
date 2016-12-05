import logging as log
import numpy as np
import os
from keras.models import Sequential
from keras.layers import TimeDistributed, Bidirectional, LSTM, Dense, Activation, Masking
from experiments.sequencenet import SequenceNet, train
from experiments.marking.tags import CategoricalTags
from experiments.marking.encoder import SentenceEncoder
from experiments.data_common import unpickle


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

        train(net, data, epoch_size=1024, nbepochs=10, nb_val_samples=128)
    else:
        log.warning('ClassifierNet training: file {} with data was not found! Please, create dataset first.')
