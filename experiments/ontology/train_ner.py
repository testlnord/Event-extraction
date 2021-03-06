import json
import logging as log
import pathlib
import random

import spacy
from spacy.gold import GoldParse
from spacy.pipeline import EntityRecognizer

from experiments.dl_utils import print_confusion_matrix
from experiments.ontology.symbols import NEW_ENT_CLASSES, ENT_CLASSES, ALL_ENT_CLASSES


def train_ner(_nlp, train_data, iterations, learn_rate=1e-3, dropout=0., tags_complete=True):
    """
    Train spacy entity recogniser (either the new on or update existing _nlp.entity)
    :param _nlp: spacy.lang.Language class, containing EntityRecogniser which is to be trained
    :param train_data: dataset in spacy format for training
    :param iterations: num of full iterations through the dataset
    :param learn_rate:
    :param dropout:
    :param tags_complete: if True, then assume that provided entity tags are complete
    :return:
    """
    _nlp.entity.model.learn_rate = learn_rate
    for itn in range(1, iterations+1):
        random.shuffle(train_data)
        loss = 0.
        for old_doc, entity_offsets in train_data:
            doc = _nlp.make_doc(old_doc.text)  # it is needed despite that the data is already preprocessed (by _nlp() call)
            gold = GoldParse(doc, entities=entity_offsets)

            # By default, the GoldParse class assumes that the entities
            # described by offset are complete, and all other words should
            # have the tag 'O'. You can tell it to make no assumptions
            # about the tag of a word by giving it the tag '-'.
            if not tags_complete:
                for i in range(len(gold.ner)):
                    if gold.ner[i] == 'O':
                        gold.ner[i] = '-'

            _nlp.tagger(doc)  # todo: why is that? is it needed for updating existing? is it needed for new model?

            loss += _nlp.entity.update(doc, gold, drop=dropout)
        log.info('train_ner: iter #{}/{}, loss: {}'.format(itn, iterations, loss))
        if loss == 0:
            break


def save_model(_nlp, model_dir, save_vectors=True, vectors_symlink=False):
    """
    :param _nlp:
    :param model_dir:
    :param save_vectors:
    :param vectors_symlink: whether symlink to old word vectors location
    :return:
    """
    model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir()
    assert model_dir.is_dir()

    _nlp.save_to_directory(model_dir)

    if save_vectors:
        vectors_path = _nlp.path / 'vocab' / 'vec.bin'
        if vectors_path.exists():
            new_vectors_path = model_dir / 'vocab' / 'vec.bin'
            if vectors_symlink:
                new_vectors_path.symlink_to(vectors_path)
            else:
                _nlp.vocab.dump_vectors(str(new_vectors_path))
        else:
            log.warning('save_model: word vectors not found at path {}! nothing to save.'.format(str(vectors_path)))


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


def test_look(y_true, y_pred, labels=ENT_CLASSES, nil='O'):
    classes = list(sorted(labels)) #+ [nil]
    stat = print_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=classes, max_print_width=7)
    _classes = list(sorted(ALL_ENT_CLASSES)) #+ [nil]
    _cm = print_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=_classes, max_print_width=7)
    return stat


def print_dataset(dataset):
    from cytoolz import first
    for i, (sent, ents) in enumerate(dataset, 1):
        print()
        print(i, sent)
        for a, b, ent_type in sorted(ents, key=first):
            print('[{}:{}] <{}> "{}"'.format(a, b, ent_type, sent.text[a:b]))
        print(ents)


def main():
    import os
    from itertools import islice
    from experiments.data_utils import split, unpickle
    from experiments.ontology.data import transform_ner_dataset
    from experiments.ontology.config import config

    random.seed(2)
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    log.info('train_ner: starting loading...')
    nlp = spacy.load('en_core_web_md')

    ner_dir = config['data']['ner_dir']
    dataset_file = os.path.join(ner_dir, 'crecords.v3.pck')
    # dataset = list(islice(unpickle(dataset_file), 100))
    dataset = list(unpickle(dataset_file))
    dataset = list(transform_ner_dataset(nlp, dataset,
                                         allowed_ent_types=ALL_ENT_CLASSES, min_ents=2))
    random.shuffle(dataset)
    tr_data, ts_data = split(dataset, (0.8, 0.2))
    log.info('#train: {}; #test: {}'.format(len(tr_data), len(ts_data)))

    epochs = 2
    epoch_size = 5
    start_epoch = 1  # for proper model saving when continuing training
    lrs = [0.001, 0.0003, 0.0001]
    lrs.extend(lrs[-1:] * epochs)  # repeat last learn rate
    save_every = 1

    nlp2 = nlp  # loading plain spacy model to train it on our classes
    ts_trues, ts_preds = get_preds(nlp2, ts_data, print_=False)
    test_look(ts_trues, ts_preds)

    # Add yet unknown entity types
    for ent_type in NEW_ENT_CLASSES:
        nlp2.entity.add_label(ent_type)

    # Add new words to vocab
    for doc, _ in tr_data:
        for word in doc:
            _ = nlp2.vocab[word.orth]

    stat_history = []
    for epoch in range(start_epoch, epochs + start_epoch):
        lr = lrs[epoch]
        train_ner(nlp2, tr_data, iterations=epoch_size, dropout=0., learn_rate=lr, tags_complete=True)
        # This step averages the model's weights. This may or may not be good for your situation --- it's empirical.
        # nlp2.end_training()

        this_model = 'models.cls.v8.1.i{}.epoch{}'.format(epoch_size, epoch)
        if epoch % save_every == 0:
            save_model(nlp2, this_model, vectors_symlink=False)
            print('saved "{}"'.format(this_model))
        print('train_ner: "{}": finished epoch: {}/{}; lr: {}'.format(this_model, epoch, epochs, lr))

        # print("##### TRAIN DATA #####")
        # tr_trues, tr_preds = get_preds(nlp2, tr_data)
        # test_look(tr_trues, tr_preds)
        print("##### TEST DATA #####")
        ts_trues, ts_preds = get_preds(nlp2, ts_data, print_=False)
        stat_history.append(test_look(ts_trues, ts_preds))


if __name__ == '__main__':
    from experiments.ontology.data_structs import ContextRecord, EntityRecord
    main()
