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
from experiments.ontology.symbols import NEW_ENT_CLASSES, ENT_CLASSES, ALL_ENT_CLASSES


def train_ner(nlp, train_data, iterations, learn_rate=1e-3, dropout=0., tags_complete=True, train_new=False):
    """
    Train spacy entity recogniser (either the new on or update existing nlp.entity)
    :param nlp: spacy.lang.Language class, containing EntityRecogniser which is to be trained
    :param train_data: dataset in spacy format for training
    :param iterations: num of full iterations through the dataset
    :param learn_rate:
    :param dropout:
    :param tags_complete: if True, then assume that provided entity tags are complete
    :param train_new: if True, train new EntityRecogniser (not update existing)
    :return:
    """
    # todo: some troubles with train_new
    if train_new: nlp.entity = EntityRecognizer(nlp.vocab, entity_types=ALL_ENT_CLASSES)

    # Add unknown entity types
    for ent_type in NEW_ENT_CLASSES:
        nlp.entity.add_label(ent_type)

    # Add new words to vocab
    for doc, _ in train_data:
        for word in doc:
            _ = nlp.vocab[word.orth]

    nlp.entity.model.learn_rate = learn_rate
    for itn in range(1, iterations+1):
        random.shuffle(train_data)
        loss = 0.
        for old_doc, entity_offsets in train_data:
            doc = nlp.make_doc(old_doc.text)  # it is needed despite that the data is already preprocessed (by nlp() call)
            gold = GoldParse(doc, entities=entity_offsets)

            # By default, the GoldParse class assumes that the entities
            # described by offset are complete, and all other words should
            # have the tag 'O'. You can tell it to make no assumptions
            # about the tag of a word by giving it the tag '-'.
            if not tags_complete:
                for i in range(len(gold.ner)):
                    if gold.ner[i] == 'O':
                        gold.ner[i] = '-'

            if not train_new:
                nlp.tagger(doc)  # todo: why is that? is it needed for updating existing? is it needed for new model?

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


# todo: what to do with features?
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
    from sklearn.metrics import confusion_matrix
    cms = []
    classes = ENT_CLASSES + ['']
    for old_doc, entity_offsets in test_data:
        raw_doc = updated_nlp.make_doc(old_doc.text)
        gold = GoldParse(raw_doc, entities=entity_offsets)
        true_ents = [g[2:] for g in gold.ner]

        toktexts = [t.text for t in old_doc]
        raw_spacy_ents = [t.ent_type_ for t in old_doc]

        doc = updated_nlp(old_doc.text)
        pred_ents = [t.ent_type_ for t in doc]

        # Some statistics
        cm = confusion_matrix(y_true=true_ents, y_pred=pred_ents, labels=classes)
        cms.append(cm)
        trues = sum(cm[i, i] for i in range(len(cm)))

        for t, true_, pos, s, S in zip(toktexts, true_ents, trues, raw_spacy_ents, pred_ents):
            print(t.ljust(20), true_.ljust(14), int(pos), s.ljust(14), S.ljust(14))
        print('Confusion matrix:', classes)
        print(cm)
        print()
    all_cm = sum(cms)
    print('Confusion matrix:', classes)
    print(all_cm)


def main():
    from experiments.ontology.data import load_superclass_mapping
    from experiments.data_utils import split
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    log.info('train_ner: starting...')
    dataset_dir = '/home/user/datasets/dbpedia/ner/'
    dataset_file = 'crecords.pck'
    dataset = list(unpickle(dataset_dir + dataset_file))
    sclasses = load_superclass_mapping()
    dataset = list(transform_ner_dataset(nlp, dataset[:], allowed_ent_types=ENT_CLASSES, superclasses_map=sclasses))
    tr_data, ts_data = split(dataset, (0.9, 0.1))

    log.info('train_ner: starting training...')
    train_ner(nlp, tr_data, iterations=100, dropout=0.5, learn_rate=0.001, tags_complete=True, train_new=False)

    model_dir = 'models2'
    save_model(nlp, model_dir)
    # nlp2 = spacy.load('en', path=model_dir)
    # test_look(nlp2, ts_data)

    test_look(nlp, ts_data)


if __name__ == '__main__':
    from experiments.ontology.data import ContextRecord, EntityRecord  # for unpickle()
    main()
