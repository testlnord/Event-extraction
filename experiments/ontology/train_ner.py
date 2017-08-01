import logging as log
import json
import pathlib
import random

import spacy
from spacy.pipeline import EntityRecognizer
from spacy.gold import GoldParse
from spacy.tagger import Tagger

from experiments.data_utils import unpickle
from experiments.ontology.data import transform_ner_dataset, nlp
from experiments.ontology.symbols import ENT_CLASSES


def train_ner(nlp, train_data, iterations, learn_rate=1e-3, dropout=0., tags_complete=False):
    """
    Train (update, actually) spacy entity recogniser
    :param nlp: spacy.lang.Language class, containing EntityRecogniser which is to be trained
    :param train_data: dataset in spacy format for training
    :param iterations: num of full iterations through the dataset
    :param learn_rate:
    :param dropout:
    :param tags_complete: if True, then assume that provided entity tags are complete
    :return:
    """
    # Add new words to vocab
    for doc, _ in train_data:
        for word in doc:
            _ = nlp.vocab[word.orth]

    # ner = EntityRecognizer(nlp.vocab, entity_types=entity_types)  # for the full training of zero-state EntityRecognizer
    # Add unknown entity types
    for ent_type, spacy_ent_type in ENT_CLASSES.items():
        if spacy_ent_type is None:
            nlp.entity.add_label(ent_type)

    # You may need to change the learning rate. It's generally difficult to
    # guess what rate you should set, especially when you have limited data.
    nlp.entity.model.learn_rate = learn_rate
    for itn in range(1, iterations+1):
        random.shuffle(train_data)
        loss = 0.
        for doc, entity_offsets in train_data:
            doc = nlp.make_doc(doc.text)  # todo: is it needed? data is preprocessed by nlp() call, actually
            gold = GoldParse(doc, entities=entity_offsets)

            # By default, the GoldParse class assumes that the entities
            # described by offset are complete, and all other words should
            # have the tag 'O'. You can tell it to make no assumptions
            # about the tag of a word by giving it the tag '-'.
            if not tags_complete:
                for i in range(len(gold.ner)):
                    # if not gold.ner[i].endswith('ANIMAL'):
                    if gold.ner[i] == 'O':
                        gold.ner[i] = '-'

            nlp.tagger(doc)  # make predictions
            # As of 1.9, spaCy's parser now lets you supply a dropout probability
            # This might help the model generalize better from only a few examples.
            loss += nlp.entity.update(doc, gold, drop=dropout)
        log.info('train_ner: iter #{}/{}, loss: {}'.format(itn, iterations, loss))
        if loss == 0:
            break
    # This step averages the model's weights. This may or may not be good for your situation --- it's empirical.
    nlp.end_training()


def save_model(nlp, model_dir):
    model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir()
    assert model_dir.is_dir()

    nlp.save_to_directory(model_dir)

    ner = nlp.entity
    with (model_dir / 'config.json').open('wb') as file_:
        data = json.dumps(ner.cfg)
        if isinstance(data, str):
            data = data.encode('utf8')
        file_.write(data)
    ner.model.dump(str(model_dir / 'model'))
    if not (model_dir / 'vocab').exists():
        (model_dir / 'vocab').mkdir()
    ner.vocab.dump(str(model_dir / 'vocab' / 'lexemes.bin'))
    with (model_dir / 'vocab' / 'strings.json').open('w', encoding='utf8') as file_:
        ner.vocab.strings.dump(file_)


# todo: what is nlp.tagger? what to do with features?
def some(nlp):
    # v1.1.2 onwards
    if nlp.tagger is None:
        print('---- WARNING ----')
        print('Data directory not found')
        print('please run: `python -m spacy.en.download --force all` for better performance')
        print('Using feature templates for tagging')
        print('-----------------')
        nlp.tagger = Tagger(nlp.vocab, features=Tagger.feature_templates)

    example_train_data = [
        (
            'Who is Shaka Khan?',
            [(len('Who is '), len('Who is Shaka Khan'), 'PERSON')]
        ),
        (
            'I like London and Berlin.',
            [(len('I like '), len('I like London'), 'LOC'),
             (len('I like London and '), len('I like London and Berlin'), 'LOC')]
        )
    ]


def test_look(updated_nlp, test_data):
    nums_errors = []
    lengths = []
    for doc, entity_offsets in test_data:
        gold = GoldParse(doc, entities=entity_offsets)
        true_ents = gold.ner
        texts = [t.text for t in doc]
        raw_spacy_ents = [t.ent_iob_+'-'+t.ent_type_ for t in doc]
        updated_nlp.tagger(doc)  # modify in-place
        retagged_ents = [t.ent_iob_+'-'+t.ent_type_ for t in doc]
        positives = [t.ent_type_ == g[2:] for t, g in zip(doc, gold.ner)]
        num_errors = len(doc) - sum(positives)
        nums_errors.append(num_errors)
        lengths.append(len(doc))
        for t, true_, pos, s, S in zip(texts, true_ents, positives, raw_spacy_ents, retagged_ents):
            print(t.ljust(20), true_.ljust(12), int(pos), s.ljust(12), S.ljust(12))
        print('errors: {}; error_ratio: {}'.format(num_errors, num_errors / len(doc)))
    print(nums_errors)
    print(lengths)
    total = sum(nums_errors)
    total_ratio = total / sum(lengths) if sum(lengths) > 0 else 0
    print('TOTAL ERRORS: {}; TOTAL_ERROR_RATIO: {}'.format(total, total_ratio))


if __name__ == '__main__':
    from experiments.ontology.data import ContextRecord, EntityRecord  # for unpickle()
    from experiments.ontology.data import load_superclass_mapping
    from experiments.data_utils import split
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    log.info('train_ner: starting...')
    dataset_dir = '/home/user/datasets/dbpedia/ner/'
    dataset_file = 'crecords.pck'
    dataset = list(unpickle(dataset_dir + dataset_file))
    sclasses = load_superclass_mapping()
    dataset = list(transform_ner_dataset(nlp=nlp, crecords=dataset[:], superclasses_map=sclasses))
    tr_data, ts_data = split(dataset, (0.9, 0.1))

    log.info('train_ner: starting training...')
    train_ner(nlp, tr_data, iterations=100, dropout=0., learn_rate=0.01, tags_complete=True)

    save_model(nlp, model_dir='models')

    test_look(nlp, ts_data)

