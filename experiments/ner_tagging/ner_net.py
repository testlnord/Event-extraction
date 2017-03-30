import logging as log
import re
from itertools import cycle

import numpy as np
from keras.layers import TimeDistributed, Bidirectional, LSTM, Dense, Masking
from keras.models import Sequential
from spacy.tokens import Span

from experiments.data_utils import split
from experiments.ner_tagging.ner_fetcher import NERDataFetcher
from experiments.sequencenet import SequenceNet

# For testing and training
from spacy.en import English
from experiments.ner_tagging.ngram_encoder import LetterNGramEncoder
from experiments.tags import CategoricalTags


class NERNet(SequenceNet):
    def compile_model(self):
        # Architecture from the original paper (arxiv.org/pdf/1608.06757.pdf):
        # Dense(150) + Bidir(LSTM(20)) + LSTM(20) + Dense(3 classes) without dropout
        dense1_output_len = 160
        lstm1_cell_size = 32
        lstm2_cell_size = 32
        # dense1_output_len = 256
        # lstm1_cell_size = 64
        # lstm2_cell_size = 64
        # lstm_dropout = 0.5
        # lstm_rdropout = 0.5

        m = Sequential()
        m.add(Masking(input_shape=(self.timesteps, self.x_len)))
        m.add(TimeDistributed(Dense(dense1_output_len, activation='tanh'),
                              input_shape=(self.timesteps, self.x_len)))
        m.add(Bidirectional(LSTM(lstm1_cell_size, return_sequences=True,
                                 # dropout=lstm_dropout, recurrent_dropout=lstm_rdropout,
                                 ), merge_mode='concat'))
        m.add(LSTM(lstm2_cell_size, return_sequences=True,
                                 # dropout=lstm_dropout, recurrent_dropout=lstm_rdropout,
                                 ))
        m.add(TimeDistributed(Dense(self.nbclasses, activation='softmax')))

        m.compile(loss='categorical_crossentropy', optimizer='adam', sample_weight_mode='temporal')
        self._model = m

    def __call__(self, doc):
        """Accepts spacy.Doc and modifies it in place (sets IOB tags for tokens)"""
        spacy_doc_classes = [token.ent_iob for token in doc]
        doc_classes = []

        # Predict IOB classes
        for sents_batch in self.batcher.batch(doc.sents):
            classes_batch = self.predict_batch(sents_batch)
            for classes in classes_batch:
                doc_classes.extend(classes)

        assert len(doc_classes) == len(spacy_doc_classes)

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
        entities_regex = re.compile('31*|1+')
        doc_classes_str = ''.join(map(str, spacy_doc_classes))
        spans = [m.span() for m in entities_regex.finditer(doc_classes_str)]
        entity_spans = (doc[a:b] for a, b in spans)
        # Usual words have ent_type = 0, so any entity will have ent_type > 0, therefore max(...)
        spacy_ent_types = [max(token.ent_type for token in span) for span in entity_spans]
        # Construct entities (i.e. Spans) preserving spacy entity types
        entities = tuple(Span(doc=doc, start=start, end=end, label=ent_type)
                         for (start, end), ent_type in zip(spans, spacy_ent_types))

        # Finally, modify doc in-place
        doc.ents = entities

    def predict_batch(self, tokenized_texts):
        orig_lengths = [len(ttext) for ttext in tokenized_texts]
        texts_enc = [self.padder.pad(self._encoder.encode_data(ttext))[0] for ttext in tokenized_texts]
        x = np.array(texts_enc)
        batch_size = len(x)

        classes_batch = self._model.predict_classes(x, batch_size=batch_size)
        # Cut to original lengths
        classes_batch = [classes[:orig_length] for orig_length, classes in zip(orig_lengths, classes_batch)]
        return classes_batch


class SpacyTokenizer:
    """
    Only for training NER Network (because NgramEncoder needs spacy.Token inputs)
    """
    def __init__(self, nlp):
        self.nlp = nlp

    def __call__(self, data_gen):
        for data, tag in data_gen:
            yield self.encode_data(data), tag

    def encode_data(self, data):
        encoded = [self.nlp(token) for token in data]
        return encoded


def nernet_sanity_check(net, nlp):
    from itertools import chain
    samples0 = [
        'Founding member Kojima Minoru played guitar on Good Day , and Wardanceis cover '
        'of a song by UK post punk industrial band Killing Joke .',
    ]

    samples_path = NERNet.relpath("data", "samples.txt")
    with open(samples_path) as filefilefile:
        samples = list(filefilefile.readlines())
        prepared_texts = [nlp(text)[:] for text in chain(samples0, samples)]
        for i, (ptext, pred) in enumerate(zip(prepared_texts, net.predict_batch(prepared_texts))):
            str_pred = [' '] * len(str(ptext))
            for token, pred_i in zip(ptext, pred):
                index = token.idx
                str_pred[index] = str(pred_i)
            print('#{}'.format(i))
            print('text:', ptext)
            print('pred:', ''.join(str_pred))


def train():
    timesteps = 150
    batch_size = 16
    epoch_steps = 1024
    epochs = 13
    val_steps = 64
    test_steps = 1024
    # epoch_size = batch_size * epoch_steps
    # nb_val_samples = batch_size * val_steps
    # nb_test_samples = batch_size * test_steps
    # nb_test_samples = 28768 # (almost?) true number of unique test samples in dataset

    tags = CategoricalTags(('O', 'I', 'B'))
    encoder = LetterNGramEncoder.from_vocab_file(tags)
    # Loading existing model
    model_path = NERNet.relpath("models", "nernet_model_{}_full_epochsize{}_epoch{}.h5".format("i0_iter1", 8192, 12))
    net = NERNet.from_model_file(encoder=encoder, batch_size=batch_size, model_path=model_path)
    # Or instantiating new model
    # net = NERNet(encoder=encoder, timesteps=timesteps, batch_size=batch_size)
    # net.compile_model()

    # Loading and splitting data
    nlp = English()
    tokenizer = SpacyTokenizer(nlp)
    fetcher = NERDataFetcher()
    data = list(fetcher.objects())
    splits = (0.1, 0.2, 0.7)
    data_splits = split(data, splits)

    # Cycle first because it is cheaper to tokenize each time than to store tokens in memory
    data_splits = [tokenizer(cycle(data_split)) for data_split in data_splits]
    data_val = data_splits[0]
    data_test = data_splits[1]
    data_train = data_splits[2]

    model_prefix = "i0_iter2"
    net.train(data_train_gen=data_train, steps_per_epoch=epoch_steps, epochs=epochs,
              data_val_gen=data_val, validation_steps=val_steps, model_prefix=model_prefix)
    # evaluation = net.evaluate(data_test, steps=test_steps)
    # print('Evaluation: {}'.format(evaluation))


if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

    # train()
    # exit()

    # Some simple testing
    tags = CategoricalTags(('O', 'I', 'B'))
    encoder = LetterNGramEncoder.from_vocab_file(tags)
    path = NERNet.relpath("models","nernet_model_{}_full_epochsize{}_epoch{}.h5".format("i0_iter2", 16384, 12))
    path = None  # loading default model
    net = NERNet.from_model_file(encoder=encoder, batch_size=4, model_path=path)
    # That way (lading weights separately from model) we can adjust number of timesteps
    # net = NERNet(encoder, timesteps=100, batch_size=8)
    # net.compile_model()
    # net.load_weights()

    nlp = English()
    nernet_sanity_check(net, nlp)

