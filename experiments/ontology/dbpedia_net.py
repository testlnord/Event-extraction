import logging as log
from itertools import cycle

import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Concatenate, TimeDistributed, Bidirectional, MaxPooling1D, Dropout, GlobalMaxPooling1D

from experiments.sequencenet import SequenceNet


class DBPediaNet(SequenceNet):
    def get_model2old(self, dr=0.5, rdr=0.5):
        input_lens = self._encoder.vector_length
        min_units=16
        max_units = 128

        inputs = [Input(shape=(None, ilen)) for ilen in input_lens]
        # Add embedding if the input length is too large
        embedded = [TimeDistributed(Dense(max_units, activation='sigmoid'))(inputs[i])
                    if input_lens[i] > max_units else inputs[i] for i in range(len(inputs))]
        # Make more units for smaller channels and cut too large channels
        xlens = [max(min_units, min(max_units, xlen)) for xlen in input_lens]

        # todo: add inputs from previous final_dense to next layer's lstm?
        lstms1 = [LSTM(xlen, return_sequences=True)(input) for input, xlen in zip(embedded, xlens)]
        lstms2 = [LSTM(xlen, return_sequences=True, dropout=dr, recurrent_dropout=rdr)(lstm1) for lstm1, xlen in zip(lstms1, xlens)]
        lstms3 = [LSTM(xlen, return_sequences=True, dropout=dr, recurrent_dropout=rdr)(lstm2) for lstm2, xlen in zip(lstms2, xlens)]

        all_lstms = [lstms1, lstms2, lstms3]
        all_aux_lstms = [[LSTM(xlen, return_sequences=False, dropout=dr, recurrent_dropout=rdr)(lstm) for lstm, xlen in zip(lstms, xlens)] for lstms in all_lstms]

        # todo: add dropout to dense?
        auxs = [Concatenate()(aux_lstms) for aux_lstms in all_aux_lstms]

        return inputs, auxs

    def get_model2(self, dr=0.5, rdr=0.5):
        input_lens = self._encoder.vector_length
        min_units = 32
        # max_units = 128
        max_units = 9999

        inputs = [Input(shape=(None, ilen)) for ilen in input_lens]
        # Add embedding if the input length is too large
        embedded = [TimeDistributed(Dense(max_units, activation='sigmoid'))(inputs[i])
                    if input_lens[i] > max_units else inputs[i] for i in range(len(inputs))]
        # Make more units for smaller channels and cut too large channels
        xlens = [max(min_units, min(max_units, xlen)) for xlen in input_lens]
        total_units = sum(xlens)

        lstms1 = [LSTM(xlen, return_sequences=True)(input) for input, xlen in zip(embedded, xlens)]
        lstms2 = [LSTM(xlen, return_sequences=True, dropout=dr, recurrent_dropout=rdr)(lstm1) for lstm1, xlen in zip(lstms1, xlens)]
        lstms3 = [LSTM(xlen, return_sequences=True, dropout=dr, recurrent_dropout=rdr)(lstm2) for lstm2, xlen in zip(lstms2, xlens)]

        aux_lstms = [Concatenate()(lstms) for lstms in [lstms1, lstms2, lstms3]]
        # aux_lstms = [MaxPooling1D(pool_size=2)(mlstm) for mlstm in aux_lstms]
        aux_lstms = [LSTM(total_units, return_sequences=False, dropout=dr, recurrent_dropout=rdr)(lstm) for lstm in aux_lstms]
        # aux_lstms = [MaxPooling1D(pool_size=2)(lstm) for lstm in aux_lstms]

        return inputs, aux_lstms

    def compile2(self, aux_dense_units=256, dr=0.5, rdr=0.5):
        inputs, lasts = self.get_model2(dr, rdr)

        lasts = [Dense(aux_dense_units, activation='sigmoid')(l) for l in lasts]
        last = Concatenate()(lasts)
        last = Dropout(rate=dr)(last)
        output = Dense(self.nbclasses, activation='softmax', name='output')(last)

        self._model = Model(inputs=inputs, outputs=[output])
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def compile3(self, dr=0.5, rdr=0.5):
        input_lens = self._encoder.vector_length
        min_units = 32
        # max_units = 128
        max_units = 9999

        inputs = [Input(shape=(None, ilen)) for ilen in input_lens]
        # Add embedding if the input length is too large
        embedded = [TimeDistributed(Dense(max_units, activation='sigmoid'))(inputs[i])
                    if input_lens[i] > max_units else inputs[i] for i in range(len(inputs))]
        # Make more units for smaller channels and cut too large channels
        xlens = [max(min_units, min(max_units, xlen)) for xlen in input_lens]
        total_units = sum(xlens)

        lstms1 = [LSTM(xlen, return_sequences=True)(input) for input, xlen in zip(embedded, xlens)]
        # lstms1 = [Bidirectional(LSTM(xlen, return_sequences=True))(input) for input, xlen in zip(embedded, xlens)]
        mlstm1 = Concatenate()(lstms1)
        # mlstm1 = MaxPooling1D(pool_size=2)(mlstm1)
        mlstm2 = LSTM(total_units // 2, return_sequences=True, dropout=dr, recurrent_dropout=rdr)(mlstm1)
        # mlstm2 = Bidirectional(LSTM(total_units // 2, return_sequences=True, dropout=dr, recurrent_dropout=rdr))(mlstm1)
        # mlstm2 = MaxPooling1D(pool_size=2)(mlstm2)
        # mlstm3 = LSTM(total_units // 2, return_sequences=False, dropout=dr, recurrent_dropout=rdr)(mlstm2)
        mlstm3 = Bidirectional(LSTM(total_units // 2, return_sequences=False, dropout=dr, recurrent_dropout=rdr))(mlstm2)

        last = mlstm3
        last = Dropout(rate=dr)(last)
        output = Dense(self.nbclasses, activation='softmax', name='output')(last)

        self._model = Model(inputs=inputs, outputs=[output])
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def compile4(self, aux_dense_units=256, dr=0.5, rdr=0.5):
        """Compile 2 subnetworks (for two branches of shprtest dependency path)"""
        inputs1, left_auxs = self.get_model2(dr, rdr)
        inputs2, right_auxs = self.get_model2(dr, rdr)
        inputs = inputs1 + inputs2

        assert(len(left_auxs) == len(right_auxs))
        aux_levels = [Concatenate()(list(pair)) for pair in zip(left_auxs, right_auxs)]

        denses = [Dense(aux_dense_units, activation='sigmoid')(aux_level) for aux_level in aux_levels]
        last = Concatenate()(denses)
        output = Dense(self.nbclasses, activation='softmax', name='output')(last)

    def compile5(self):
        xlens = self._encoder.vector_length

        last_units = 256

        inputs = [Input(shape=(None, xlen)) for xlen in xlens]
        lstms1 = [LSTM(xlen, return_sequences=True)(input) for input, xlen in zip(inputs, xlens)]
        lstms2 = [LSTM(xlen, return_sequences=True)(lstm1) for lstm1, xlen in zip(lstms1, xlens)]
        lstms3 = [LSTM(xlen, return_sequences=True)(lstm2) for lstm2, xlen in zip(lstms2, xlens)]
        last = Concatenate()(lstms3)  # todo: what axis, really? there're two axes for each layer: length and xlen.

        # todo: adjust pool size to be a divisor of lstms' output lengths
        maxpools1 = [TimeDistributed(MaxPooling1D(pool_size=2))(lstm1) for lstm1 in lstms1]
        maxpool1 = Concatenate()(maxpools1)

        all_lstms = [lstms1, lstms2, lstms3]
        all_maxpools = [[TimeDistributed(MaxPooling1D(pool_size=2))(lstm) for lstm in lstms] for lstms in all_lstms]
        cmaxpools = [Concatenate()(maxpools) for maxpools in all_maxpools]


        # dense = TimeDistributed(Dense(last_units, activation='softmax'))(last)
        dense = Dense(last_units, activation='softmax')(last)

        output = dense
        self._model = Model(inputs=inputs, outputs=[output])
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', sample_weight_mode='temporal')  # sample_weight_mode??

    def _make_data_gen(self, data_gen):
        # data_gen = self.padder(xy for data in data_gen for xy in self._encoder.encode(data))
        data_gen = self._encoder(data_gen)
        c = self._encoder.channels  # so, arr[:c] is the inputs and the arr[c:] is the output(s)
        res = ((arr[:c], arr[c:]) for arr in self.batcher.batch_transposed(data_gen))  # decouple inputs and outputs (classes)
        return cycle(res)

    def predict(self, subject_object_spans_pairs, topn=1):
        encoded = [self._encoder.encode_data(*so_pair) for so_pair in subject_object_spans_pairs]
        preds = []
        for batch in self.batcher.batch_transposed(encoded):
            preds_batch = self._model.predict_on_batch(batch)
            preds.extend(preds_batch)
        all_tops = []
        decode = self._encoder.tags.decode
        for pred in preds:
            top_inds = np.argsort(-pred)[:topn]
            probs = pred[top_inds]
            classes = [decode(ind) for ind in top_inds]
            tops = list(zip(top_inds, probs, classes))
            all_tops.append(tops)
        return all_tops if topn > 1 else sum(all_tops, [])  # remove extra dimension

    # temporary method for testing
    def predict_crecords(self, crecords, topn=1):
        c = self._encoder.channels
        encoded = [arr[:c] for cr in crecords for arr in self._encoder.encode(cr)]  # throwing out the true class
        preds = []
        for batch in self.batcher.batch_transposed(encoded):
            preds_batch = self._model.predict_on_batch(batch)
            preds.extend(preds_batch)
        all_tops = []
        decode = self._encoder.tags.decode
        for pred in preds:
            top_inds = np.argsort(-pred)[:topn]
            probs = pred[top_inds]
            classes = [decode(ind) for ind in top_inds]
            tops = list(zip(top_inds, probs, classes))
            all_tops.append(tops)
        return all_tops if topn > 1 else sum(all_tops, [])  # remove extra dimension

    # this method is specific for output of form [categorical, binary]
    def predict_dir(self, subject_object_spans_pairs, topn=1, direction_threshold=0.5):
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
            # todo: decode dir classes (arg_max etc.)
            # direction = int(direction[0] >= direction_threshold)
            # direction = dirdecode(direction)
            # dirs = [direction] * topn
            top_inds = np.argsort(-pred)[:topn]
            probs = pred[top_inds]
            classes = [decode(ind) for ind in top_inds]
            tops = list(zip(top_inds, probs, classes))
            all_tops.append(tops)
        return all_tops if topn > 1 else sum(all_tops, [])  # remove extra dimension

