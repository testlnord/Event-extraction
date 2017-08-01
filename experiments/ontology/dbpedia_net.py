import logging as log
from itertools import cycle

import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Concatenate, TimeDistributed, Bidirectional, MaxPool1D, Dropout

from experiments.data_utils import split, visualise
from experiments.sequencenet import SequenceNet
from experiments.ontology.ont_encoder import DBPediaEncoder, DBPediaEncoderBranched


class DBPediaNet(SequenceNet):
    def compile1(self):
        xlens = self._encoder.vector_length

        dr = 0.5
        rdr = 0.5

        inputs = [Input(shape=(None, xlen)) for xlen in xlens]
        lstms1 = [LSTM(xlen, return_sequences=True)(input) for input, xlen in zip(inputs, xlens)]
        lstms2 = [LSTM(xlen, return_sequences=True, dropout=dr, recurrent_dropout=rdr)(lstm1) for lstm1, xlen in zip(lstms1, xlens)]
        lstms3 = [LSTM(xlen, return_sequences=False, dropout=dr, recurrent_dropout=rdr)(lstm2) for lstm2, xlen in zip(lstms2, xlens)]

        # todo: add dropout to dense?
        last = Concatenate(axis=-1)(lstms3)
        output = Dense(self.nbclasses, activation='softmax')(last)

        self._model = Model(inputs=inputs, outputs=[output])
        # self._model.compile(loss='categorical_crossentropy', optimizer='adam', sample_weight_mode='temporal')  # sample_weight_mode??
        self._model.compile(loss='categorical_crossentropy', optimizer='adam')

    def get_model2(self, min_units=16, aux_dense_units=256, dr=0.5, rdr=0.5):
        input_lens = self._encoder.vector_length

        inputs = [Input(shape=(None, ilen)) for ilen in input_lens]
        xlens = [max(min_units, xlen) for xlen in input_lens]  # more units for simpler channels
        # todo: add inputs from previous final_dense to next layer's lstm?
        lstms1 = [LSTM(xlen, return_sequences=True)(input) for input, xlen in zip(inputs, xlens)]
        lstms2 = [LSTM(xlen, return_sequences=True, dropout=dr, recurrent_dropout=rdr)(lstm1) for lstm1, xlen in zip(lstms1, xlens)]
        lstms3 = [LSTM(xlen, return_sequences=True, dropout=dr, recurrent_dropout=rdr)(lstm2) for lstm2, xlen in zip(lstms2, xlens)]

        all_lstms = [lstms1, lstms2, lstms3]
        # aux_lstms2 = [LSTM(xlen, return_sequences=False, dropout=dr, recurrent_dropout=rdr)(lstm2) for lstm2, xlen in zip(lstms2, xlens)]
        all_aux_lstms = [[LSTM(xlen, return_sequences=False, dropout=dr, recurrent_dropout=rdr)(lstm) for lstm, xlen in zip(lstms, xlens)] for lstms in all_lstms]

        # todo: add dropout to dense?
        auxs = [Concatenate()(aux_lstms) for aux_lstms in all_aux_lstms]

        return inputs, auxs

    def compile2(self, min_units=16, aux_dense_units=256, dr=0.5, rdr=0.5):
        inputs, auxs = self.get_model2(min_units, aux_dense_units, dr, rdr)

        denses = [Dense(aux_dense_units, activation='sigmoid')(aux) for aux in auxs]
        last = Concatenate()(denses)
        output = Dense(self.nbclasses, activation='softmax', name='output')(last)

        # Multiple outputs
        direction_output = Dense(1, activation='sigmoid', name='direction_output')(last)
        self._model = Model(inputs=inputs, outputs=[output, direction_output])
        self._model.compile(optimizer='adam',
                            loss={'output': 'categorical_crossentropy', 'direction_output': 'binary_crossentropy'},
                            loss_weights={'output': 1, 'direction_output': 1})

        # Single output
        # self._model.compile(loss='categorical_crossentropy', optimizer='adam', sample_weight_mode='temporal')  # sample_weight_mode??
        # self._model = Model(inputs=inputs, outputs=[output])
        # self._model.compile(loss='categorical_crossentropy', optimizer='adam')

    def compile4(self, min_units=16, aux_dense_units=256, dr=0.5, rdr=0.5):
        """Compile 2 subnetworks (for two branches of shprtest dependency path)"""
        inputs1, left_auxs = self.get_model2(min_units, aux_dense_units, dr, rdr)
        inputs2, right_auxs = self.get_model2(min_units, aux_dense_units, dr, rdr)
        inputs = inputs1 + inputs2

        assert(len(left_auxs) == len(right_auxs))
        aux_levels = [Concatenate()(list(pair)) for pair in zip(left_auxs, right_auxs)]

        denses = [Dense(aux_dense_units, activation='sigmoid')(aux_level) for aux_level in aux_levels]
        last = Concatenate()(denses)
        output = Dense(self.nbclasses, activation='softmax', name='output')(last)

        # Multiple outputs
        direction_output = Dense(1, activation='sigmoid', name='direction_output')(last)
        self._model = Model(inputs=inputs, outputs=[output, direction_output])
        self._model.compile(optimizer='adam',
                            loss={'output': 'categorical_crossentropy', 'direction_output': 'binary_crossentropy'},
                            loss_weights={'output': 1, 'direction_output': 1})

    def compile3(self):
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
        c = self._encoder.channels
        # todo: if len(arr[c:]) == 1 then use arr[c] ??? is it necessary
        res = ((arr[:c], arr[c:]) for arr in self.batcher.batch_transposed(data_gen))  # decouple inputs and outputs (classes)
        return res

    # this method is specific for output of form [categorical, binary]
    def predict(self, subject_object_spans_pairs, topn=1, direction_threshold=0.5):
        encoded = [self._encoder.encode_data(*so_pair) for so_pair in subject_object_spans_pairs]
        preds = []
        directions = []
        for batch in self.batcher.batch_transposed(encoded):
            preds_batch, directions_batch = self._model.predict_on_batch(batch)
            preds.extend(preds_batch)
            directions.extend(directions_batch)
        all_tops = []
        decode = self._encoder.tags.decode
        for pred, direction in zip(preds, directions):  # todo: check for multiple outputs
            direction = int(direction[0] >= direction_threshold)
            dirs = [direction] * topn
            top_inds = np.argsort(-pred)[:topn]
            probs = pred[top_inds]
            classes = [decode(ind) for ind in top_inds]
            tops = list(zip(top_inds, probs, classes, dirs))
            all_tops.append(tops)
        return all_tops if topn > 1 else sum(all_tops, [])  # remove extra dimension


def eye_test(net, crecords):
    test_data = [crecord2spans(crecord, nlp) for crecord in crecords]
    for tops, crecord in zip(net.predict(test_data, topn=3), crecords):
        print()
        print(crecord.context)
        true_dir = int(crecord.s_end <= crecord.o_start)
        _ = list(net._encoder.encode(crecord))  # to get the sdp
        print(net._encoder.last_sdp)
        print(true_dir, crecord.triple)
        for icls, prob, rel, direction in tops:
            print('{:2d} {:.2f} {} {}'.format(icls, prob, direction, rel))


if __name__ == "__main__":
    # np.random.seed(2)

    # import spacy
    # nlp = spacy.load('en')  # it is imported from other files for now
    from experiments.ontology.data import nlp, classes_dir, load_prop_superclass_mapping, load_inverse_mapping, load_all_data
    from experiments.ontology.ont_encoder import crecord2spans

    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    batch_size = 1
    prop_case = 'test.balanced0.'
    model_name = prop_case + 'dr.noaug.v3'

    scls_file = classes_dir + 'prop_classes.{}csv'.format(prop_case)
    sclasses = load_prop_superclass_mapping(scls_file)
    inv_file = classes_dir + 'prop_inverse.csv'
    inverse = load_inverse_mapping(inv_file)
    dataset = load_all_data(sclasses, shuffle=True)
    train_data, val_data = split(dataset, splits=(0.8, 0.2), batch_size=batch_size)
    log.info('total: {}; train: {}; val: {}'.format(len(dataset), len(train_data), len(val_data)))

    epochs = 6
    train_steps = len(train_data) // batch_size
    val_steps = len(val_data) // batch_size

    encoder = DBPediaEncoder(nlp, sclasses, inverse, augment_data=False, expand_noun_chunks=False)
    # encoder = DBPediaEncoderBranched(nlp, sclasses, inverse, augment_data=False, expand_noun_chunks=False)
    net = DBPediaNet(encoder, timesteps=None, batch_size=batch_size)
    net.compile2()
    model_path = 'dbpedianet_model_{}_full_epochsize{}_epoch{:02d}.h5'.format(model_name, train_steps, 3)
    # net = DBPediaNet.from_model_file(encoder, batch_size, model_path=DBPediaNet.relpath('models', model_path))

    log.info('model: {}; epochs: {}'.format(model_name, epochs))
    net._model.summary(line_length=80)

    # eye_test(net, val_data)

    net.train(cycle(train_data), epochs, train_steps, cycle(val_data), val_steps, model_prefix=model_name)
    log.info('end training')

    # evals = net.evaluate(cycle(val_data), val_steps)
    # print(evals)
    # tests = [619, 1034, 1726, 3269, 6990(6992?)]  # some edge cases


