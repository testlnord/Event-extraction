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


def get_preds(updated_nlp, test_data, nil='O', print_=False):
    y_true = []
    y_pred = []
    for old_doc, entity_offsets in test_data:
        raw_doc = updated_nlp.make_doc(old_doc.text)
        gold = GoldParse(raw_doc, entities=entity_offsets)
        true_ents = [g[2:] if g[2:] else nil for g in gold.ner]
        y_true.extend(true_ents)
        doc = updated_nlp(old_doc.text)
        pred_ents = [t.ent_type_ if t.ent_type_ else nil for t in doc]
        y_pred.extend(pred_ents)
        if print_:
            toktexts = [t.text for t in old_doc]
            raw_spacy_ents = [t.ent_type_ if t.ent_type_ else nil for t in old_doc]
            for d in zip(toktexts, true_ents, raw_spacy_ents, pred_ents):
                print(''.join([_.ljust(20) for _ in d]))
    return y_true, y_pred


def print_confusion_matrix(cm, labels):
    from numpy import vectorize, vstack, set_printoptions
    fp = cm[:-1, -1].sum()
    fn = cm[-1, :-1].sum()
    tp = sum([cm[i, i] for i in range(len(cm)-1)])
    tn = cm[-1, -1]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    l = max([len(str(x)) for x in cm.flatten()])
    printer = vectorize(lambda x: str(x)[:l].rjust(l))
    set_printoptions(linewidth=240)
    # print(printer(vstack((labels, cm))))
    print('  ' + ' '.join(printer(labels)))
    print(cm)
    print('tn: {}; tp: {}; fp: {}; fn: {}'.format(tn, tp, fp, fn))
    print('precision: {}; recall: {}; f1: {}'.format(precision, recall, f1))
    print()


def test_look(y_true, y_pred, labels=ENT_CLASSES, nil='O'):
    from sklearn.metrics import confusion_matrix
    classes = labels + [nil]
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=classes)
    print_confusion_matrix(cm, classes)
    _classes = ALL_ENT_CLASSES + [nil]
    _cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=_classes)
    print_confusion_matrix(_cm, _classes)
    return cm


def main():
    from experiments.data_utils import split
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    log.info('train_ner: starting...')
    dataset_dir = '/home/user/datasets/dbpedia/ner/'
    dataset_file = 'crecords.v2.pck'
    model_dir = 'models_v2'

    dataset = list(unpickle(dataset_dir + dataset_file))
    dataset = list(transform_ner_dataset(nlp, dataset[:], allowed_ent_types=ALL_ENT_CLASSES))
    tr_data, ts_data = split(dataset, (0.9, 0.1))

    # log.info('train_ner: starting training...')
    # train_ner(nlp, tr_data, iterations=100, dropout=0., learn_rate=0.001, tags_complete=True, train_new=False)

    # save_model(nlp, model_dir)
    nlp2 = spacy.load('en', path=model_dir)
    print("##### TRAIN DATA #####")
    tr_trues, tr_preds = get_preds(nlp2, tr_data)
    print("##### TEST DATA #####")
    ts_trues, ts_preds = get_preds(nlp2, ts_data)
    print("##### TRAIN DATA #####")
    test_look(tr_trues, tr_preds)
    print("##### TEST DATA #####")
    test_look(ts_trues, ts_preds)


if __name__ == '__main__':
    from experiments.ontology.data import ContextRecord, EntityRecord  # for unpickle()
    main()
