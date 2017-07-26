import logging as log
from itertools import cycle

import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Concatenate, TimeDistributed, Bidirectional, MaxPool1D, Dropout

from experiments.data_utils import split, visualise
from experiments.sequencenet import SequenceNet
from experiments.ontology.ont_encoder import DBPediaEncoder


class DBPediaNet(SequenceNet):
    def compile1(self):
        xlens = self._encoder.vector_length

        dr = 0.4
        rdr = 0.4

        inputs = [Input(shape=(None, xlen)) for xlen in xlens]
        lstms1 = [LSTM(xlen, return_sequences=True)(input) for input, xlen in zip(inputs, xlens)]
        lstms2 = [LSTM(xlen, return_sequences=True, dropout=dr, recurrent_dropout=rdr)(lstm1) for lstm1, xlen in zip(lstms1, xlens)]
        lstms3 = [LSTM(xlen, return_sequences=False, dropout=dr, recurrent_dropout=rdr)(lstm2) for lstm2, xlen in zip(lstms2, xlens)]
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

    def _make_data_gen(self, data_gen):
        # data_gen = self.padder(xy for data in data_gen for xy in self._encoder.encode(data))
        data_gen = self._encoder(data_gen)
        res = ((arr[:-1], arr[-1]) for arr in self.batcher.batch_transposed(data_gen))  # decouple inputs and output (classes)
        # res = ((arr[:-1], np.reshape(arr[-1], (self.batch_size, 1, -1))) for arr in self.batcher.batch_transposed(data_gen))
        return res

    def predict(self, subject_object_spans_pairs, topn=1):
        encoded = [self._encoder.encode_data(*so_pair) for so_pair in subject_object_spans_pairs]
        preds = []
        for batch in self.batcher.batch_transposed(encoded):
            preds_batch = self._model.predict_on_batch(batch)
            preds.extend(preds_batch)
        all_tops = []
        decode = self._encoder.tags.decode
        for pred in preds:
            # top_inds = np.argpartition(pred, -topn)[-topn:]  # see: https://stackoverflow.com/a/23734295
            # probs = np.sort(pred[top_inds])  # not quite right
            top_inds = np.argsort(-pred)[:topn]
            probs = pred[top_inds]
            classes = [decode(ind) for ind in top_inds]
            tops = list(zip(top_inds, probs, classes))
            all_tops.append(tops)
        return all_tops if topn > 1 else sum(all_tops, [])  # remove extra dimension


def eye_test(net, crecords):
    test_data = [crecord2spans(crecord, nlp) for crecord in crecords]
    for tops, crecord in zip(net.predict(test_data, topn=3), crecords):
        print()
        print(crecord.triple)
        for icls, prob, rel in tops:
            print('{:2d} {:.2f} {}'.format(icls, prob, rel))


if __name__ == "__main__":
    # import spacy
    # nlp = spacy.load('en')  # it is imported from other files for now
    from experiments.ontology.sub_ont import nlp
    from experiments.ontology.data import props_dir, load_superclass_mapping, load_inverse_mapping, load_all_data
    from experiments.ontology.ont_encoder import crecord2spans

    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)
    # np.random.seed(2)

    batch_size = 1
    prop_case = 'test.balanced0.'
    model_name = prop_case + 'dr.noaug.v1.'

    scls_file = props_dir + 'prop_classes.{}csv'.format(prop_case)
    sclasses = load_superclass_mapping(scls_file)
    inv_file = props_dir + 'prop_inverse.csv'
    inverse = load_inverse_mapping(inv_file)
    encoder = DBPediaEncoder(nlp, sclasses, inverse)
    dataset = load_all_data(sclasses, shuffle=True)
    train_data, val_data = split(dataset, splits=(0.8, 0.2), batch_size=batch_size)
    print('total: {}; train: {}; val: {}'.format(len(dataset), len(train_data), len(val_data)))

    epochs = 6
    train_steps = len(train_data) // batch_size
    val_steps = len(val_data) // batch_size

    encoder = DBPediaEncoder(nlp, sclasses, inverse, augment_data=False)
    net = DBPediaNet(encoder, timesteps=None, batch_size=batch_size)
    net.compile1()
    # model_path = 'dbpedianet_model_{}_full_epochsize{}_epoch{:02d}.h5'.format(model_name, train_steps, 2)
    # net = DBPediaNet.from_model_file(encoder, batch_size, model_path=DBPediaNet.relpath('models', model_path))
    net._model.summary(line_length=80)

    # eye_test(net, val_data)

    net.train(cycle(train_data), epochs, train_steps, cycle(val_data), val_steps, model_prefix=model_name)
    # evals = net.evaluate(cycle(val_data), val_steps)
    # print(evals)
    # tests = [619, 1034, 1726, 3269, 6990(6992?)]  # some edge cases


