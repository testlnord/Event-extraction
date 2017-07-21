import logging as log

from keras.models import Model
from keras.layers import Dense, LSTM, Input, Concatenate, TimeDistributed, Bidirectional, MaxPool1D

from experiments.sequencenet import SequenceNet


class DBPediaNet(SequenceNet):
    def compile1(self):
        xlens = self._encoder.nbclasses

        last_units = 256

        inputs = [Input(shape=(None, xlen)) for xlen in xlens]
        lstms1 = [LSTM(xlen, return_sequences=True)(input) for input, xlen in zip(inputs, xlens)]
        lstms2 = [LSTM(xlen, return_sequences=True)(lstm1) for lstm1, xlen in zip(lstms1, xlens)]
        lstms3 = [LSTM(xlen, return_sequences=False)(lstm2) for lstm2, xlen in zip(lstms2, xlens)]
        last = Concatenate()(lstms3)
        dense = Dense(last_units, activation='softmax')(last)

        output = dense
        model = Model(inputs=inputs, outputs=[output])
        model.compile(loss='categorical_crossentropy', optimizer='adam', sample_weight_mode='temporal')  # sample_weight_mode??

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
        model = Model(inputs=inputs, outputs=[output])
        model.compile(loss='categorical_crossentropy', optimizer='adam', sample_weight_mode='temporal')  # sample_weight_mode??

if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)
    pass
