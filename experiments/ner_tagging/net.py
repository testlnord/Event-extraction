import logging as log
import numpy as np
import re
from itertools import cycle
from spacy.tokens import Span
from keras.models import Sequential, load_model
from keras.layers import TimeDistributed, Bidirectional, LSTM, Dense, Activation, Masking
from experiments.sequencenet import SequenceNet
from experiments.data_common import *
from experiments.ner_tagging.preprocessor import NERPreprocessor


class NERNet(SequenceNet):
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
        # texts_enc = list(self.padder(tuple(self._encoder(tokenized_texts))))
        x = np.array(texts_enc)
        batch_size = len(x)

        classes_batch = self._model.predict_classes(x, batch_size=batch_size)
        # cut to original lengths
        classes_batch = [classes[:orig_length] for orig_length, classes in zip(orig_lengths, classes_batch)]
        return classes_batch


def eye_test(nlp):
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


        "Google will be dropping its own implementation of Java 's APIs in Android N in favour of the open-source OpenJDK .",

        "Version 1.0 of JetBrains' Kotlin programmatic language is now available for JVM and Android developers .",

        "Microsoft has released a new mobile version of Windows 10 for developers subscribing to the Windows Insider program .",

        "Adobe has also released version 21.0.0.176 of AIR Desktop Runtime , AIR SDK , "
        "AIR SDK & Compiler and AIR for Android , which contain Flash Player components .",

    ]

    trues = [
        'B O O O O O B O O O B I I I O',
        'O O B I O O O B I O O B O O O O O B O O O O B I O',
        'O B I O O O O O O O O B O O O O O B I I I O B O O O O O O O O O O O',
        'B I O O O O B I O O B O O B O O O O O O O O B O O O O O O O O',
        'O B I I O O O O O O O O O O O O O O O O O O O B O B I I I I O O B O O O B O O',
        '',
        '',
        '',
        '',
    ]

    texts2 = [
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
    for i, (text, true, pred) in enumerate(zip(texts, trues, net.predict_batch(prepared_texts))):
        print('#{}'.format(i))
        print('text:', text)
        print('pred:', pred)
        print('true: ', true)

    prepared_texts = [nlp(text)[:] for text in texts2]
    for i, (text, pred) in enumerate(zip(texts2, net.predict_batch(prepared_texts))):
        print('#{}'.format(i))
        print('text:', text)
        print('pred:', pred)


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

    from experiments.main_pipeline import load_nlp, load_default_ner_net
    from spacy.en import English
    # nlp = load_nlp()
    nlp = English()
    net = load_default_ner_net(batch_size=batch_size)

    # tags = CategoricalTags(('O', 'I', 'B'))
    # vocab_path = 'models/encoder_vocab_{}gram_{}len.bin'.format(3, vector_length)
    # encoder = LetterNGramEncoder.from_vocab_file(nlp, tags)
    # model_path = 'models/model_full_epochsize{}_epoch{:02d}_valloss{}.h5'.format(8192, 8, 0.23)
    # model_path = 'models/model_full_epochsize{}_epoch{}.h5'.format(16384, 6)
    # net = NERNet.from_model_file(encoder=encoder, batch_size=batch_size, model_path=model_path)
    # train(net)

    eye_test(nlp)

