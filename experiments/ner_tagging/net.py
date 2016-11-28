import logging as log
import numpy as np
import os
import re
from itertools import cycle
from keras.models import Sequential, load_model
from keras.layers import TimeDistributed, Bidirectional, LSTM, Dense, Activation, Masking
from keras.callbacks import ModelCheckpoint, TensorBoard
from experiments.data_common import *
from experiments.marking.tags import CategoricalTags
from experiments.ner_tagging.preprocessor import NERPreprocessor
from experiments.ner_tagging.encoder import LetterNGramEncoder


class NERNet:
    def __init__(self, encoder, timesteps, batch_size=8):
        self.x_len = encoder.vector_length
        self.batch_size = batch_size
        self.nbclasses = encoder.tags.nbtags
        self.timesteps = timesteps
        self._encoder = encoder
        self._model = None

        self.padder = Padding(pad_to_length=timesteps)
        self.batcher = BatchMaker(self.batch_size)

    @classmethod
    def from_model_file(cls, encoder, model_path=None, batch_size=8):
        if not model_path:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            model_name = 'nernet_model_full_default.h5'
            model_path = os.path.join(model_dir, model_name)

        model = load_model(model_path)
        _, timesteps, x_len = model.layers[0].input_shape

        if x_len != encoder.vector_length:
            raise ValueError('NERNet: encoder.vector_length is not consistent with the input of '
                             'loaded model ({} != {})'.format(encoder.vector_length, x_len))

        net = cls(encoder, timesteps, batch_size)
        net._model = model
        return net

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
        self._model = m

    def _make_data_gen(self, data_gen, do_infinite=True):
        # Cycling for keras
        if do_infinite: data_gen = cycle(data_gen)

        data_gen = self._encoder(data_gen)
        if self.timesteps:
            data_gen = self.padder(data_gen)
        return self.batcher.batch_transposed(data_gen)

    def train(self, data_train_gen, epochs, epoch_size,
              data_val_gen, nb_val_samples,
              save_epoch_models=True, dir_for_models='./models',
              log_for_tensorboard=True, dir_for_logs='./logs'):
        """Train model, maybe with checkpoints for every epoch and logging for tensorboard"""
        log.info('NERNet: Training...')

        data_train = self._make_data_gen(data_train_gen)
        data_val = self._make_data_gen(data_val_gen)
        callbacks = []
        # ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2)

        if save_epoch_models:
            filepath = dir_for_models + '/model_full_epochsize{}'.format(epoch_size) + '_epoch{epoch:02d}_valloss{val_loss:.2f}.h5'
            mcheck_cb = ModelCheckpoint(filepath, verbose=0, save_weights_only=False, mode='auto')
            callbacks.append(mcheck_cb)
        if log_for_tensorboard:
            tb_cb = TensorBoard(log_dir=dir_for_logs, histogram_freq=1, write_graph=True, write_images=False)
            callbacks.append(tb_cb)

        return self._model.fit_generator(data_train,
                                         samples_per_epoch=epoch_size, nb_epoch=epochs,
                                         max_q_size=2,
                                         validation_data=data_val, nb_val_samples=nb_val_samples,
                                         callbacks=callbacks
                                         )

    def save_model(self, path):
        self._model.save(path)

    def load_model(self, path):
        log.info('NERNet: Loading model ({})...'.format(path))
        del self._model
        self._model = load_model(path)
        log.info('NERNet: Loaded model.')

    def evaluate(self, data_gen, nb_val_samples):
        log.info('NERNet: Evaluating...')
        data_val = self._make_data_gen(data_gen)
        return self._model.evaluate_generator(data_val, max_q_size=2, val_samples=nb_val_samples)

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
        texts_enc = [self.padder.pad(self._encoder.encode_text(ttext))[0] for ttext in tokenized_texts]
        x = np.array(texts_enc)
        batch_size = len(x)

        classes_batch = self._model.predict_classes(x, batch_size=batch_size)
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


def train(net):
    data_thing = NERPreprocessor()
    splits = (0.1, 0.2, 0.7)
    data_splits = split(data_thing, splits)
    data_val = data_splits[0]
    data_test = data_splits[1]
    data_train = data_splits[2]

    # Training
    epoch_size = 8192
    nbepochs = 13
    hist = net.train(data_train, nbepochs, epoch_size, data_val, 1024)
    print('Hisory:', hist.history)

    # Saving history
    hist_path = './logs/history_epochsize{}_epochs{}.json'.format(epoch_size, nbepochs)
    with open(hist_path, 'w') as f:
        import json
        json.dump(hist.history, f)

    # Evaluating
    print('Evaluation: {}'.format(net.evaluate(data_test)))

if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

    vector_length = 30000
    batch_size = 16
    model_path = 'models/model_full_epochsize{}_epoch{:02d}_valloss{}.h5'.format(8192, 8, 0.23)
    # model_path = 'models/model_full_epochsize{}_epoch{}.h5'.format(16384, 6)

    from spacy.en import English
    nlp = English()
    tags = CategoricalTags(('O', 'I', 'B'))
    vocab_path = 'models/encoder_vocab_{}gram_{}len.bin'.format(3, vector_length)
    encoder = LetterNGramEncoder.from_vocab_file(nlp, tags, vocab_path)

    # Loading model
    net = NERNet.from_model_file(encoder, model_path, batch_size)

    # train(net)
    eye_test()

