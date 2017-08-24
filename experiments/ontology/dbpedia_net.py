import logging as log
from itertools import cycle

import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Concatenate, Embedding, \
    Dropout, TimeDistributed, Bidirectional, MaxPooling1D, GlobalMaxPooling1D
from keras import regularizers

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
        max_units = 512

        inputs = [Input(shape=(None, ilen)) for ilen in input_lens]
        # Make more units for smaller channels and cut too large channels
        xlens = [max(min_units, min(max_units, xlen)) for xlen in input_lens]
        total_units = sum(xlens)

        # Original compile3
        lstms1 = [LSTM(xlen, return_sequences=True)(input) for input, xlen in zip(inputs, xlens)]
        mlstm1 = Concatenate()(lstms1)
        # compile3.v2 -- not good?
        # embedded = Concatenate()(embedded)
        # mlstm1 = LSTM(total_units, return_sequences=True, dropout=dr, recurrent_dropout=rdr)(embedded)

        mlstm2 = LSTM(total_units // 2, return_sequences=True, dropout=dr, recurrent_dropout=rdr)(mlstm1)
        # mlstm2 = Bidirectional(LSTM(total_units // 2, return_sequences=True, dropout=dr, recurrent_dropout=rdr))(mlstm1)
        # mlstm3 = LSTM(total_units // 2, return_sequences=False, dropout=dr, recurrent_dropout=rdr)(mlstm2)
        mlstm3 = Bidirectional(LSTM(total_units // 2, return_sequences=False, dropout=dr, recurrent_dropout=rdr))(mlstm2)

        last = mlstm3
        last = Dropout(rate=dr)(last)
        output = Dense(self.nbclasses, activation='softmax', name='output')(last)

        self._model = Model(inputs=inputs, outputs=[output])
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def get_model3(self, input_lens, dr=0., rdr=0., reversed_inputs=False, lstm_layers=2):
        embed_size = 50
        # echannels = set(self._encoder.embedding_channels)
        echannels = list(range(len(input_lens) - 2))

        # todo: embedding for word vectors and preloading of spacy word vectors
        inputs = [Input(shape=(None, ), dtype='int32') if i in echannels else
                  Input(shape=(None, ilen)) for i, ilen in enumerate(input_lens)]

        embedded = [Embedding(ilen, embed_size)(inp) if i in echannels else inp for i, (ilen, inp) in enumerate(zip(input_lens, inputs))]
        embedded = [Dropout(rate=dr)(emb) if i in echannels else emb for i, emb in enumerate(embedded)]  # dropout embeddings
        xlens = [embed_size if i in echannels else ilen for i, ilen in enumerate(input_lens)]

        nexts = embedded
        if reversed_inputs:
            nexts = list(reversed(nexts))
            xlens = list(reversed(xlens))

        for i in range(lstm_layers):
            # nexts = [LSTM(xlen, return_sequences=True)(l) for xlen, l in zip(xlens, nexts)]
            nexts = [LSTM(xlen, return_sequences=True, dropout=dr, recurrent_dropout=rdr)(l) for xlen, l in zip(xlens, nexts)]
            # nexts = [Bidirectional(LSTM(xlen, return_sequences=True, dropout=dr, recurrent_dropout=rdr))(l) for xlen, l in zip(xlens, nexts)]

        pooled = [GlobalMaxPooling1D()(l) for l in nexts]
        return inputs, pooled

    def compile4b(self, dr=0.5, rdr=0.5, l2=0.):
        last_dense = 100
        rl2 = regularizers.l2(l2)

        input_lens = self._encoder.vector_length
        c = len(input_lens) // 2  # splitting evenly, as branches are identical in their features
        left_lens, right_lens = input_lens[:c], input_lens[c:]  # see EncoderBranched.vector_length

        # todo: try non-reversed
        linputs, lefts = self.get_model3(left_lens, dr, rdr)
        rinputs, rights = self.get_model3(right_lens, dr, rdr, reversed_inputs=True)

        channels = [Concatenate()([l, r]) for l, r in zip(lefts, rights)]  # concat left & right subpath in each channel
        # channels = [Dense(last_dense, activation='sigmoid', kernel_regularizer=rl2)(ch) for ch in channels]  # final channel representation

        last = Concatenate()(channels)  # all channels merged; final representation
        last = Dense(last_dense, activation='sigmoid', kernel_regularizer=rl2)(last)
        last = Dropout(rate=dr)(last)

        inputs = linputs + rinputs
        output = Dense(self.nbclasses, activation='softmax', kernel_regularizer=rl2, name='output')(last)
        self._model = Model(inputs=inputs, outputs=[output])
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def compile4(self, dr=0.5, rdr=0.5, l2=0.):
        last_dense = 200
        rl2 = regularizers.l2(l2)

        input_lens = self._encoder.vector_length
        inputs, channels = self.get_model3(input_lens, dr, rdr)

        last = Concatenate()(channels)  # all channels merged; final representation
        last = Dense(last_dense, activation='sigmoid', kernel_regularizer=rl2)(last)
        last = Dropout(rate=dr)(last)

        output = Dense(self.nbclasses, activation='softmax', kernel_regularizer=rl2, name='output')(last)
        self._model = Model(inputs=inputs, outputs=[output])
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def _make_data_gen(self, data_gen):
        # data_gen = self.padder(xy for data in data_gen for xy in self._encoder.encode(data))
        c = self._encoder.channels  # so, arr[:c] is the inputs and the arr[c:] is the output(s)
        data_gen = self._encoder(data_gen)
        res = ((arr[:c], arr[c:]) for arr in self.batcher.batch_transposed(data_gen))  # decouple inputs and outputs (classes)
        return cycle(res)

    def predict(self, subject_object_spans_pairs, topn=1):
        encoded = [self._encoder.encode_data(*so_pair) for so_pair in subject_object_spans_pairs]
        predictions = self._predict_encoded(encoded, topn)
        assert len(predictions) == len(encoded) == len(subject_object_spans_pairs)  # need to keep consistency: predictions must be aligned with data
        return predictions

    # temporary method for testing
    def predict_crecords(self, crecords, topn=1):
        c = self._encoder.channels
        encoded = [self._encoder.encode(cr)[:c] for cr in crecords]
        predictions = self._predict_encoded(encoded, topn)
        assert len(predictions) == len(encoded) == len(crecords)  # need to keep consistency: predictions must be aligned with data
        return predictions

    def _predict_encoded(self, encoded, topn):
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
        return all_tops
        # return all_tops if topn > 1 else sum(all_tops, [])  # remove extra dimension
