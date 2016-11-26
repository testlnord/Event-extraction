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
            self.compile_model()

    def data(self, splits=(0.1, 0.2, 0.7)):
        data_length = len(self._data_thing)
        edges = split_range(data_length, self.batch_size, splits)
        self.data_splits = []

        for a, b in edges:
            data_split_gen_func = None
            if self.timesteps:
                def data_split():
                    return self.batch(self.padder(
                        self.encoder(islice(self._data_thing.objects(), a, b))))
                data_split_gen_func = data_split
            else:
                def data_split():
                    return self.batch(
                        self.encoder(islice(self._data_thing.objects(), a, b)))
                data_split_gen_func = data_split

            self.data_splits.append(cycle_uncached(data_split_gen_func))

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

    def compile_model(self):
        # architecture is from the paper
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
        self.model.fit_generator(self.data_train,
                                 samples_per_epoch=epoch_size, nb_epoch=epochs,
                                 max_q_size=2,
                                 validation_data=self.data_val, nb_val_samples=nb_val_samples,
                                 )

    def train2(self, epochs, epoch_size=16384, nb_val_samples=512):
        log.info('NERNet: Training...')
        for ii, i in enumerate(range(0, epochs * epoch_size, epoch_size), 1):
            print('EPOCH {}/{}'.format(ii, epochs), '[{}, {})'.format(i, i+epoch_size))
            # self.train(1, epoch_size, nb_val_samples)
            self.model.fit_generator(self.data_train,
                                     samples_per_epoch=epoch_size, nb_epoch=1,
                                     max_q_size=2,
                                     validation_data=self.data_val, nb_val_samples=nb_val_samples,
                                     )
            self.save_model(path='model_full_epochsize{}_epoch{}.h5'.format(epoch_size, ii))

    def save_model(self, path='model_full.h5'):
        self.model.save(path)

    def load_model(self, path='model_full.h5'):
        log.info('NERNet: Loading model ({})...'.format(path))
        del self.model
        self.model = load_model(path)
        log.info('NERNet: Loaded model.')

    # todo: ValueError: generator already executing (e.g. when executing after training)
    def evaluate(self, nb_val_samples):
        log.info('NERNet: Evaluating...')
        return self.model.evaluate_generator(self.data_val, max_q_size=2, val_samples=nb_val_samples)

    # todo: test
    def __call__(self, doc):
        """Accepts spacy.Doc and modifies it in place (sets IOB tags for tokens)"""
        for sents_batch in self.batch(doc.sents):
            classes_batch = self.predict_batch(sents_batch)
            # decoding categorial values (classes) to original tags and converting numpy int to normal python int
            tags_batch = [self.encoder.decode_tags(map(int, classes)) for classes in classes_batch]
            for sent, sent_tags, sent_classes in zip(sents_batch, tags_batch, classes_batch):
                for token, iob_tag, iob_class in zip(sent, sent_tags, sent_classes):
                    token.ent_iob = iob_class
                    token.ent_iob_ = iob_tag

    def predict_batch(self, tokenized_texts):
        orig_lengths = [len(ttext) for ttext in tokenized_texts]
        texts_enc = [self.padder.pad(self.encoder.encode_text(ttext))[0] for ttext in tokenized_texts]
        x = np.array(texts_enc)
        batch_size = len(x)

        classes_batch = self.model.predict_classes(x, batch_size=batch_size)
        # cutting to original lengths
        classes_batch = (classes[:orig_length] for orig_length, classes in zip(orig_lengths, classes_batch))
        return classes_batch


def test():
    texts = [
        '010 is the tenth album from Japanese Punk Techno band The Mad Capsule Markets .',

        'Founding member Kojima Minoru played guitar on Good Day , and Wardanceis cover '
        'of a song by UK post punk industrial band Killing Joke .',

        'The Node.js Foundation , which has jurisdiction over the popular server-side JavaScript platform , '
        'is adding the Express MVC Web framework for Node.js as an incubation project , to ensure its continued viability .',

        'PyPy 5.0 also has an upgraded C-level API so that Python scripts using C components '
        '( for example , by way of  Cython ) are both more compatible and faster .',

        'The Ruby on Rails team released versions 4.2.5.1 , 4.1.14.1 , and 3.2.22.1 of the framework last week '
        'to address multiple issues in Rails and rails-html-sanitizer , a Ruby gem that sanitizes HTML input .',
    ]
    trues = [
        'B O O O O O B O O O B I I I O',
        'O O B I O O O B I O O B O O O O O B O O O O B I O',
        'O B I O O O O O O O O B O O O O O B I I I O B O O O O O O O O O O O',
        'B I O O O O B I O O B O O B O O O O O O O O B O O O O O O O O',
        'O B I I O O O O O O O O O O O O O O O O O O O B O B O O B O O O B O O',
    ]
    prepared_texts = [np.array(text.split()) for text in texts]

    for i, (text, true, pred) in enumerate(zip(texts, trues, net.predict_batch(prepared_texts))):
        print('#{}'.format(i))
        print('text:', text)
        print('pred:', pred)
        print('true:', true)


# todo: handle History objects from train func
if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
    last_epoch = 6
    model_path = 'model_full_epochsize{}_epoch{}.h5'.format(16384, last_epoch)
    net = NERNet(x_len=30000, batch_size=16, nbclasses=3, timesteps=150, path_to_model=model_path)

    # nbepochs = 6
    # net.train2(nbepochs)
    # print('Evaluation: {}'.format(net.evaluate()))

    # for i in range(nbepochs):
    #     net.train(epochs=1, epoch_size=512, nb_val_samples=128)
    #     log.info('Training: trained epoch #{}'.format(i+1))
        # net.save_model()
        # net.load_model(model_path)

    test()



