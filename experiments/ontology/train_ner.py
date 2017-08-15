import json
import logging as log
import pathlib
import random

from spacy.gold import GoldParse
from spacy.pipeline import EntityRecognizer

from experiments.dl_utils import print_confusion_matrix
from experiments.ontology.symbols import NEW_ENT_CLASSES, ENT_CLASSES, ALL_ENT_CLASSES


def train_ner(_nlp, train_data, iterations, learn_rate=1e-3, dropout=0., tags_complete=True, train_new=False):
    """
    Train spacy entity recogniser (either the new on or update existing _nlp.entity)
    :param _nlp: spacy.lang.Language class, containing EntityRecogniser which is to be trained
    :param train_data: dataset in spacy format for training
    :param iterations: num of full iterations through the dataset
    :param learn_rate:
    :param dropout:
    :param tags_complete: if True, then assume that provided entity tags are complete
    :param train_new: if True, train new EntityRecogniser (not update existing)
    :return:
    """
    # todo: some troubles with train_new
    if train_new: _nlp.entity = EntityRecognizer(_nlp.vocab, entity_types=ALL_ENT_CLASSES)

    # Add unknown entity types
    for ent_type in NEW_ENT_CLASSES:
        _nlp.entity.add_label(ent_type)

    # Add new words to vocab
    for doc, _ in train_data:
        for word in doc:
            _ = _nlp.vocab[word.orth]

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

            if not train_new:
                _nlp.tagger(doc)  # todo: why is that? is it needed for updating existing? is it needed for new model?

            loss += _nlp.entity.update(doc, gold, drop=dropout)
        log.info('train_ner: iter #{}/{}, loss: {}'.format(itn, iterations, loss))
        if loss == 0:
            break
    # This step averages the model's weights. This may or may not be good for your situation --- it's empirical.
    _nlp.end_training()


def save_model(_nlp, model_dir):
    model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir()
    assert model_dir.is_dir()

    _nlp.save_to_directory(model_dir)

    ner = _nlp.entity
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
    classes = list(sorted(labels)) + [nil]
    cm = print_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=classes, max_print_width=7)
    _classes = list(sorted(ALL_ENT_CLASSES)) + [nil]
    _cm = print_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=_classes, max_print_width=7)
    return cm


def main():
    from experiments.data_utils import split, unpickle
    from experiments.ontology.data import transform_ner_dataset, nlp

    random.seed(2)
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    log.info('train_ner: starting loading...')
    dataset_dir = '/home/user/datasets/dbpedia/ner/'
    dataset_file = 'crecords.v2.pck'

    # dataset = list(islice(unpickle(dataset_dir + dataset_file), 400))
    dataset = list(unpickle(dataset_dir + dataset_file))
    dataset = list(transform_ner_dataset(nlp, dataset,
                                         allowed_ent_types=ALL_ENT_CLASSES, min_ents=20, min_ents_ratio=0.05))
    tr_data, ts_data = split(dataset, (0.9, 0.1))
    # ts_data = dataset
    log.info('#train: {}; #test: {}'.format(len(tr_data), len(ts_data)))

    nlp2 = nlp
    # nlp2 = spacy.load('en', path=model_dir)  # continuing training

    epochs = 2
    iterations = 20
    for epoch in range(1, epochs + 1):
        train_ner(nlp2, tr_data, iterations=iterations, dropout=0., learn_rate=0.001, tags_complete=True, train_new=False)
        model_dir = 'models.v5.i{}.epoch{}'.format(iterations, epoch)
        save_model(nlp2, model_dir)

        print("##### TRAIN DATA #####")
        tr_trues, tr_preds = get_preds(nlp2, tr_data)
        test_look(tr_trues, tr_preds)
        print("##### TEST DATA #####")
        ts_trues, ts_preds = get_preds(nlp2, ts_data, print_=False)
        test_look(ts_trues, ts_preds)


if __name__ == '__main__':
    from experiments.ontology.data_structs import ContextRecord, EntityRecord
    main()
