import logging as log
import os
import random
from itertools import cycle

from experiments.dl_utils import print_confusion_matrix
from experiments.data_utils import unpickle, split, visualise
from experiments.ontology.ont_encoder import *
from experiments.ontology.dbpedia_net import DBPediaNet


def eye_test(net, crecords, prob_threshold=0.5):
    misses = []
    hits = []
    trues = []
    preds = []
    for tops, crecord in zip(net.predict_crecords(crecords, topn=3), crecords):
        _ = net._encoder.encode(crecord)  # to get the sdp
        sdp = net._encoder.last_sdp
        true_rel_with_dir = net._encoder.encode_raw_class(crecord)
        _struct = (crecord, sdp, tops, true_rel_with_dir)
        # if any(prob >= prob_threshold and true_rel_with_dir == rel_with_dir for icls, prob, rel_with_dir in tops):
        i, prob, rel_with_dir = tops[0]
        if true_rel_with_dir == rel_with_dir and prob >= prob_threshold:
            hits.append(_struct)
        else:
            misses.append(_struct)
        trues.append(true_rel_with_dir)
        preds.append(tops[0][2])

    print("\n### HITS ({}):".format(len(hits)), '#' * 40)
    for _struct in hits:
        print_tested(*_struct)
    print("\n### MISSES ({}):".format(len(misses)), '#' * 40)
    for _struct in misses:
        print_tested(*_struct)

    def format_class(rel_with_dir):
        return str(rel_with_dir[0])[:10] + '-' + str(rel_with_dir[1])[:1]
    _tags = net._encoder.tags
    raw_classes = list(sorted(_tags.raw_tags)) + [_tags.default_tag]
    trues = list(map(format_class, trues))
    preds = list(map(format_class, preds))
    classes = list(map(format_class, raw_classes))
    print_confusion_matrix(y_true=trues, y_pred=preds, labels=classes, max_print_width=20)

    return hits, misses


def print_tested(crecord, sdp, tops, true_rel_with_dir):
    print()
    print(crecord.context)
    print(sdp)
    print(str(crecord.subject), (str(crecord.relation), crecord.direction), str(crecord.object))
    # print(str(crecord.subject_text), (str(crecord.relation), crecord.direction), str(crecord.object_text))
    print('------>', true_rel_with_dir)
    for icls, prob, rel_with_dir in tops:
        print('{:2d} {:.2f} {}'.format(icls, prob, rel_with_dir))


def get_dbp_data(sclasses, neg_ratio, batch_size=1):
    from experiments.ontology.data import load_rc_data
    from experiments.ontology.tagger import load_golden_data

    data_dir = '/home/user/datasets/dbpedia/'
    golden_dir = '/home/user/datasets/dbpedia/rc/golden500/'
    rc_out = os.path.join(data_dir, 'rc', 'rrecords.v2.filtered.pck')
    rc0_out = os.path.join(data_dir, 'rc', 'rrecords.v2.negative.pck')

    # Load golden-set (test-data), cutting it; load train-set, excluding golden-set from there
    # golden = load_golden_data(sclasses, golden_dir, shuffle=True)[:4000]  # for testing
    # exclude = golden
    exclude = set()
    dataset = load_rc_data(sclasses, rc_file=rc_out, rc_neg_file=rc0_out, neg_ratio=neg_ratio, shuffle=True, exclude_records=exclude)
    # train_data, val_data = dataset, golden  # using golden set as testing set
    train, val = split(dataset, splits=(0.8, 0.2), batch_size=batch_size)  # usual data load

    return train, val


def load_benchmark_data(sclasses, data_dir, filenames=('train.pck', 'test.pck')):
    _classes = set(sclasses)
    _classes.update({'no_relation', 'Other', None})  # possible negative classes
    for filename in filenames:
        input_path = os.path.join(data_dir, filename)
        data = [record for record in unpickle(input_path) if record.relation in _classes]
        random.shuffle(data)
        yield data


def get_semeval_data(sclasses):
    data_dir = '/home/user/datasets/'
    data_dir = os.path.join(data_dir, 'semeval2010')
    return tuple(load_benchmark_data(sclasses, data_dir))


def get_kbp37_data(sclasses):
    data_dir = '/home/user/datasets/'
    data_dir = os.path.join(data_dir, 'kbp37')
    return tuple(load_benchmark_data(sclasses, data_dir))


def map_relations(records, mapping):
    mapping[None] = None  # preserving negative class
    new_records = []
    for record in records:
        if record.relation in mapping:
            record.relation = mapping[record.relation]
            new_records.append(record)
    return new_records


def get_all_data(sclasses, neg_ratio):
    from experiments.ontology.symbols import KBP37_CLASSES_MAPPED, SEMEVAL_CLASSES_MAPPED

    train0, val0 = get_dbp_data(sclasses, neg_ratio, 1)

    kbp_subset = KBP37_CLASSES_MAPPED
    train1, val1 = get_kbp37_data(kbp_subset)
    train1 = map_relations(train1, kbp_subset)
    val1 = map_relations(val1, kbp_subset)

    train = [train0 + train1]
    val = [val0 + val1]
    random.shuffle(train)
    random.shuffle(val)

    return train, val


def main():
    from experiments.ontology.symbols import RC_CLASSES_MAP, RC_CLASSES_MAP_ALL, RC_INVERSE_MAP
    from experiments.ontology.symbols import KBP37_CLASSES_MAP, SEMEVAL_CLASSES_MAP

    import spacy
    from experiments.ontology.config import config
    nlp = spacy.load(**config['models']['nlp'])
    # model_dir = 'models.v5.4.i5.epoch2'
    # nlp = spacy.load('en', path=model_dir)  # todo: where're the word vectors???

    random.seed(2)
    batch_size = 1
    epochs = 6

    # Load our data
    sclasses = RC_CLASSES_MAP_ALL
    inverse = RC_INVERSE_MAP
    model_name = 'nocls.v5.4.c3.spacy.inv'
    encoder = DBPediaEncoder(nlp, sclasses, inverse_relations=inverse,
                             expand_context=3, min_entities_dist=2)
    # encoder = DBPediaEncoderEmbed(nlp, sclasses, inverse_relations=inverse)
    # encoder = DBPediaEncoderBranched(nlp, sclasses, inverse_relations=inverse)
    # encoder = DBPediaEncoderWithEntTypes(nlp, sclasses, inverse_relations=inverse)
    train_data, val_data = get_dbp_data(sclasses, neg_ratio=0.5, batch_size=batch_size)

    # Load benchmark data (kbp37)
    # sclasses = KBP37_CLASSES_MAP
    # model_name = 'noner.dr.noaug.kbp.v5.3.c3'
    # encoder = DBPediaEncoder(nlp, sclasses)
    # encoder = DBPediaEncoderEmbed(nlp, sclasses)
    # train_data, val_data = get_kbp37_data(sclasses)

    # Load benchmark data (semeval)
    # sclasses = SEMEVAL_CLASSES_MAP
    # model_name = 'noner.dr.noaug.semeval.v5.3.1.c3'
    # encoder = DBPediaEncoder(nlp, sclasses)
    # model_name = 'noner.dr.noaug.semeval.v6.3.c4b'
    # encoder = DBPediaEncoderBranched(nlp, sclasses)
    # encoder = DBPediaEncoderEmbed(nlp, sclasses)
    # train_data, val_data = get_semeval_data(sclasses)

    train_steps = len(train_data) // batch_size
    val_steps = len(val_data) // batch_size
    nb_negs_tr = len([rr for rr in train_data if str(rr.relation) not in sclasses])
    nb_negs_val = len([rr for rr in val_data if str(rr.relation) not in sclasses])
    log.info('data: train: {} (negs: {}); val: {} (negs: {})'
             .format(len(train_data), nb_negs_tr, len(val_data), nb_negs_val))

    # Instantiating new net or loading existing
    net = DBPediaNet(encoder, timesteps=None, batch_size=batch_size)
    net.compile3()
    # net.compile4(l2=1e-5)
    # model_path = 'dbpedianet_model_{}_full_epoch{:02d}.h5'.format(model_name, 5)
    # net = DBPediaNet.from_model_file(encoder, batch_size, model_path=DBPediaNet.relpath('models', model_path))

    log.info('classes: {}; model: {}; epochs: {}'.format(encoder.nbclasses, model_name, epochs))
    net._model.summary(line_length=80)

    hist = net.train(train_data, epochs, train_steps, val_data, val_steps, model_prefix=model_name)  # if the net cycles data by itself

    test_data = val_data
    prob_threshold = 0.5
    hits, misses = eye_test(net, test_data, prob_threshold=prob_threshold)
    print('rights: {}/{} with prob_threshold={}'.format(len(hits), len(test_data), prob_threshold))
    print(hist)
    # evals = net.evaluate(val_data, val_steps)  # NB: remember about cycle()
    # print('evaluated: {}'.format(evals))


if __name__ == "__main__":
    from experiments.ontology.data_structs import RelationRecord, RelRecord

    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.DEBUG)
    main()
