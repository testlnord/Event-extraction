import logging as log
import numpy as np
import re
import json
from itertools import cycle, islice
from spacy.tokens import Span
from keras.models import Sequential, load_model
from keras.layers import TimeDistributed, Bidirectional, LSTM, Dense, Activation, Masking
from keras.callbacks import ModelCheckpoint, TensorBoard
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
        self.batcher = BatchMaker(self.batch_size)

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
                    return self.batcher.batch_transposed(self.padder(
                        self.encoder(islice(self._data_thing.objects(), a, b))))
                data_split_gen_func = data_split
            else:
                def data_split():
                    return self.batcher.batch_transposed(
                        self.encoder(islice(self._data_thing.objects(), a, b)))
                data_split_gen_func = data_split

            self.data_splits.append(cycle_uncached(data_split_gen_func))

        self.data_val = self.data_splits[0]
        self.data_test = self.data_splits[1]
        self.data_train = self.data_splits[2]

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

    def train(self, epochs, epoch_size, nb_val_samples=1024,
              save_epoch_models=True, dir_for_models='./models',
              log_for_tensorboard=True, dir_for_logs='./logs'):
        """Train model, maybe with checkpoints for every epoch and logging for tensorboard"""
        log.info('NERNet: Training...')

        callbacks = []
        # ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2)

        if save_epoch_models:
            filepath = dir_for_models + '/model_full_epochsize{}'.format(epoch_size) + '_epoch{epoch:02d}_valloss{val_loss:.2f}.h5'
            mcheck_cb = ModelCheckpoint(filepath, verbose=0, save_weights_only=False, mode='auto')
            callbacks.append(mcheck_cb)
        if log_for_tensorboard:
            tb_cb = TensorBoard(log_dir=dir_for_logs, histogram_freq=1, write_graph=True, write_images=False)
            callbacks.append(tb_cb)

        return self.model.fit_generator(self.data_train,
                                        samples_per_epoch=epoch_size, nb_epoch=epochs,
                                        max_q_size=2,
                                        validation_data=self.data_val, nb_val_samples=nb_val_samples,
                                        callbacks=callbacks
                                        )

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        log.info('NERNet: Loading model ({})...'.format(path))
        del self.model
        self.model = load_model(path)
        log.info('NERNet: Loaded model.')

    # todo: ValueError: generator already executing (e.g. when executing after training)
    def evaluate(self, nb_val_samples):
        log.info('NERNet: Evaluating...')
        return self.model.evaluate_generator(self.data_val, max_q_size=2, val_samples=nb_val_samples)

    def __call__(self, doc):
        """Accepts spacy.Doc and modifies it in place (sets IOB tags for tokens)"""
        spacy_doc_classes = [token.ent_iob for token in doc]
        doc_classes = []

        # Predict IOB classes
        for sents_batch in self.batcher.batch(doc.sents):
            classes_batch = self.predict_batch(sents_batch)
            for classes in classes_batch:
                doc_classes.extend(classes)

        # Merge predicted classes and spacy iob classes
        for i, (iob_class, spacy_iob_class) in enumerate(zip(doc_classes, spacy_doc_classes)):
            # Spacy iob codes: 1="I", 2="O", 3="B". 0 means no tag is assigned

            # If we assigned iob code and spacy didn't:
            if iob_class > 0 and (spacy_iob_class == 2 or spacy_iob_class == 0):

                # Make predicted iob class consistent with spacy iob codes
                if iob_class == 2: iob_class = 3
                # elif iob_class == 0: iob_class = 2

                # Use our iob code: we may find something spacy didn't
                spacy_doc_classes[i] = iob_class
                log.debug('NERNet: tagged: spacy={}, pred={}, token={}'.format(spacy_iob_class, iob_class, doc[i].text))

        # Extract entities
        entities = re.compile('3[21]*1|31*')
        doc_classes_str = ''.join(map(str, spacy_doc_classes))
        spans = [m.span() for m in entities.finditer(doc_classes_str)]
        entities = tuple(doc[a:b] for a, b in spans)

        # Modify doc in-place
        doc.ents = entities

    def predict_batch(self, tokenized_texts):
        orig_lengths = [len(ttext) for ttext in tokenized_texts]
        texts_enc = [self.padder.pad(self.encoder.encode_text(ttext))[0] for ttext in tokenized_texts]
        x = np.array(texts_enc)
        batch_size = len(x)

        classes_batch = self.model.predict_classes(x, batch_size=batch_size)
        # cut to original lengths
        classes_batch = [classes[:orig_length] for orig_length, classes in zip(orig_lengths, classes_batch)]
        return classes_batch


def eye_test():
    texts = [
        '010 is the tenth album from Japanese Punk Techno band The Mad Capsule Markets .',

        'Founding member Kojima Minoru played guitar on Good Day , and Wardanceis cover '
        'of a song by UK post punk industrial band Killing Joke .',

        'The Node.js Foundation , which has jurisdiction over the popular server-side JavaScript platform , '
        'is adding the Express MVC Web framework for Node.js as an incubation project , to ensure its continued viability .',

        'PyPy 5.0 also has an upgraded C-level API so that Python scripts using C components '
        '( for example , by way of  Cython ) are both more compatible and faster .',

        'The Ruby on Rails team released versions 4.2.5.1 , 4.1.14.1 , and 3.2.22.1 of the framework last week '
        'to address multiple issues in Rails and rails - html - sanitizer , a Ruby gem that sanitizes HTML input .',
    ]
    trues = [
        'B O O O O O B O O O B I I I O',
        'O O B I O O O B I O O B O O O O O B O O O O B I O',
        'O B I O O O O O O O O B O O O O O B I I I O B O O O O O O O O O O O',
        'B I O O O O B I O O B O O B O O O O O O O O B O O O O O O O O',
        'O B I I O O O O O O O O O O O O O O O O O O O B O B I I I I O O B O O O B O O',
    ]
    prepared_texts = [np.array(text.split()) for text in texts]

    for i, (text, true, pred) in enumerate(zip(texts, trues, net.predict_batch(prepared_texts))):
        print('#{}'.format(i))
        print('text:', text)
        print('pred:', pred.tolist())
        print('true:', true)


if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

    # Creating new model
    # net = NERNet(x_len=30000, batch_size=16, nbclasses=3, timesteps=150)
    # Loading model
    # model_path = 'models/model_full_epochsize{}_epoch{}.h5'.format(16384, 6)
    model_path = 'models/model_full_epochsize{}_epoch{:02d}_valloss{}.h5'.format(8192, 8, 0.23)
    net = NERNet(x_len=30000, batch_size=16, nbclasses=3, timesteps=150, path_to_model=model_path)

    # Training
    # epoch_size = 8192
    # nbepochs = 13
    # hist = net.train(nbepochs, epoch_size, 1024)
    # print('Hisory:', hist.history)
    # Saving history
    # hist_path = './logs/history_epochsize{}_epochs{}.json'.format(epoch_size, nbepochs)
    # with open(hist_path, 'w') as f:
    #     json.dump(hist.history, f)

    # Evaluating
    # print('Evaluation: {}'.format(net.evaluate()))

    eye_test()



