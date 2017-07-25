import logging as log
from itertools import cycle

import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Concatenate, TimeDistributed, Bidirectional, MaxPool1D

from experiments.data_utils import split, visualise
from experiments.sequencenet import SequenceNet
from experiments.ontology.ont_encoder import DBPediaEncoder


class DBPediaNet(SequenceNet):
    def compile1(self):
        xlens = self._encoder.vector_length

        last_units = 256

        inputs = [Input(shape=(None, xlen)) for xlen in xlens]
        lstms1 = [LSTM(xlen, return_sequences=True)(input) for input, xlen in zip(inputs, xlens)]
        lstms2 = [LSTM(xlen, return_sequences=True)(lstm1) for lstm1, xlen in zip(lstms1, xlens)]
        lstms3 = [LSTM(xlen, return_sequences=False)(lstm2) for lstm2, xlen in zip(lstms2, xlens)]
        last = Concatenate(axis=-1)(lstms3)
        dense = Dense(self.nbclasses, activation='softmax')(last)

        output = dense
        self._model = Model(inputs=inputs, outputs=[output])
        # self._model.compile(loss='categorical_crossentropy', optimizer='adam', sample_weight_mode='temporal')  # sample_weight_mode??
        self._model.compile(loss='categorical_crossentropy', optimizer='adam')


    def compile2(self):
        xlens = self._encoder.nbclasses

        last_units = 256

        inputs = [Input(shape=(None, xlen)) for xlen in xlens]
        lstms1 = [LSTM(xlen, return_sequences=True)(input) for input, xlen in zip(inputs, xlens)]
        lstms2 = [LSTM(xlen, return_sequences=True)(lstm1) for lstm1, xlen in zip(lstms1, xlens)]
        lstms3 = [LSTM(xlen, return_sequences=True)(lstm2) for lstm2, xlen in zip(lstms2, xlens)]
        last = Concatenate()(lstms3)  # todo: what axis, really? there're two axes for each layer: length and xlen.

        # todo: adjust pool size to be a divisor of lstms' output lengths
        maxpools1 = [TimeDistributed(MaxPool1D(pool_size=2))(lstm1) for lstm1 in lstms1]
        maxpool1 = Concatenate()(maxpools1)

        all_lstms = [lstms1, lstms2, lstms3]
        all_maxpools = [[TimeDistributed(MaxPool1D(pool_size=2))(lstm) for lstm in lstms] for lstms in all_lstms]
        cmaxpools = [Concatenate()(maxpools) for maxpools in all_maxpools]


        # dense = TimeDistributed(Dense(last_units, activation='softmax'))(last)
        dense = Dense(last_units, activation='softmax')(last)

        output = dense
        self._model = Model(inputs=inputs, outputs=[output])
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', sample_weight_mode='temporal')  # sample_weight_mode??

    # todo:
    def _make_data_gen(self, data_gen):
        # data_gen = self.padder(self._encoder.encode(data) for data in data_gen)
        data_gen = self._encoder(data_gen)
        res = ((arr[:-1], arr[-1]) for arr in self.batcher.batch_transposed(data_gen))  # decouple inputs and output (classes)
        # res = ((arr[:-1], np.reshape(arr[-1], (self.batch_size, 1, -1))) for arr in self.batcher.batch_transposed(data_gen))
        return res



if __name__ == "__main__":
    from experiments.ontology.sub_ont import nlp
    from experiments.ontology.data import props_dir, load_classes_dict, load_all_data

    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)
    # np.random.seed(2)

    batch_size = 1
    model_name = 'test_v1'

    # import spacy
    # nlp = spacy.load('en')  # it is imported from other files for now
    prop_classes_file = props_dir + 'prop_classes.test.csv'
    classes = load_classes_dict(prop_classes_file)
    dataset = load_all_data(classes, shuffle=True)
    train_data, val_data = split(dataset, splits=(0.8, 0.2), batch_size=batch_size)
    print('total: {}; train: {}; val: {}'.format(len(dataset), len(train_data), len(val_data)))

    epochs = 4
    train_steps = len(train_data) // batch_size
    val_steps = len(val_data) // batch_size

    encoder = DBPediaEncoder(nlp, classes)
    # net = DBPediaNet(encoder, timesteps=None, batch_size=batch_size)
    # net.compile1()
    model_path = 'dbpedianet_model_{}_full_epochsize{}_epoch{:02d}.h5'.format(model_name, train_steps, 2)
    net = DBPediaNet.from_model_file(encoder, batch_size, model_path=DBPediaNet.relpath('models', model_path))

    net._model.summary(line_length=80)

    # tests = [619, 1034, 1726, 3269, 6990(6992?)]  # some edge cases
    net.train(cycle(train_data), epochs, train_steps, cycle(val_data), val_steps, model_prefix=model_name)

    evals = net.evaluate(val_data, val_steps)
    print(evals)


