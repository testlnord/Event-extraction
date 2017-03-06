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
        # Architecture is from the original paper: arxiv.org/pdf/1608.06757.pdf
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


def eye_test(net, nlp):
    texts = [
        'Founding member Kojima Minoru played guitar on Good Day , and Wardanceis cover '
        'of a song by UK post punk industrial band Killing Joke .',

        "Google will be dropping its own implementation of Java's APIs in Android N in favour of the open-source OpenJDK."
        "Version 1.0 of JetBrains' Kotlin programmatic language is now available for JVM and Android developers."
        , "Microsoft has released a new mobile version of Windows 10 for developers subscribing to the Windows Insider program."
        , "Microsoft has announced a new pricing model for its Visual Studio platform."
        , "Adobe has also released version 21.0.0.176 of AIR Desktop Runtime, AIR SDK, AIR SDK & Compiler and AIR for Android, which contain Flash Player components."
        , "Beginning with iOS 8, iPhones, iPads, and iPad, Touches are encrypted using a key derived from the user-selected passcode."
        , "Yesterday, Microsoft capitulated and  started publishing changelogs  and  release information  for Windows 10 patches."
        , "Microsoft has updated Visual Studio Taco (Tools for Apache Cordova), keying in on error-reporting and project-template improvements."
        , "An anonymous reader writes:  Ubuntu 16.04 LTS and newer will  no longer be supporting AMD's widely-used Catalyst Linux (fglrx) driver ."
        , "China's Tencent has pulled out of Windows Phone development and has criticised Microsoft's commitment to the platform."
        , "Apple's  trendy Swift language  is getting a Web framework inspired by the popular  Ruby on Rails  MVC framework."
        , "The Ruby on Rails team released versions 4.2.5.1, 4.1.14.1, and 3.2.22.1 of the framework last week to address multiple issues in Rails and rails-html-sanitizer, a Ruby gem that sanitizes HTML input."
        , "Ruby on Rails fixed six vulnerabilities in versions 3.x, 4.1.x, 4.2.x, and Rails 5.0 beta and three in rails-html-sanitizer"
          "PyPy applications can now be embedded within a C program, allowing developers to use both C and Python, regardless of which language they're most comfortable with."
        , "PyPy 5.0 also has an upgraded C-level API so that Python scripts using C components (for example, by way of  Cython ) are both more compatible and faster."
        , "Objective C has fallen out of the top ten most popular coding languages for the first time in five years, according to the Tiobe index."
        , "Node.js Foundation just released v.4 of it's popular platform – containing the latest version of Javascript's V8 engine + support for all ARM processors."
    ]

    prepared_texts = [nlp(text)[:] for text in texts]
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
    epoch_size = 8192
    epochs = 13
    nb_val_samples = 1024
    nb_test_samples = 16384
    # nb_test_samples = 28768

    tags = CategoricalTags(('O', 'I', 'B'))
    encoder = LetterNGramEncoder.from_vocab_file(tags)
    # Loading existing model
    # model_path = 'models/model_full_epochsize{}_epoch{:02d}_valloss{}.h5'.format(8192, 8, 0.23)
    # model_path=None
    # net = NERNet.from_model_file(encoder=encoder, batch_size=batch_size, model_path=model_path)
    # Or instantiating new model
    net = NERNet(encoder=encoder, timesteps=timesteps, batch_size=batch_size)
    net.compile_model()

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

    net.train(data_train_gen=data_train, epoch_size=epoch_size, epochs=epochs,
              data_val_gen=data_val, nb_val_samples=nb_val_samples)
    evaluation = net.evaluate(data_test, nb_val_samples=nb_test_samples)
    print('Evaluation: {}'.format(evaluation))


if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

    # train()
    # exit()

    tags = CategoricalTags(('O', 'I', 'B'))
    encoder = LetterNGramEncoder.from_vocab_file(tags)
    # net = NERNet.from_model_file(encoder=encoder, batch_size=8)
    net = NERNet(encoder, timesteps=100, batch_size=8)
    net.compile_model()
    net.load_weights()

    nlp = English()
    eye_test(net, nlp)

