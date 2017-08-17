import logging as log
import os
import random
from itertools import cycle

from experiments.dl_utils import print_confusion_matrix
from experiments.data_utils import unpickle, split, visualise
from experiments.ontology.ont_encoder import DBPediaEncoder, DBPediaEncoderWithEntTypes
from experiments.ontology.dbpedia_net import DBPediaNet


def eye_test(net, crecords, prob_threshold=0.5):
    misses = []
    hits = []
    trues = []
    preds = []
    for tops, crecord in zip(net.predict_crecords(crecords, topn=3), crecords):
        _ = list(net._encoder.encode(crecord))  # to get the sdp
        sdp = net._encoder.last_sdp
        true_rel_with_dir = net._encoder.encode_raw_class(crecord)
        _struct = (crecord, sdp, tops, true_rel_with_dir)
        if any(prob >= prob_threshold and true_rel_with_dir == rel_with_dir for icls, prob, rel_with_dir in tops):
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
    print('------>', true_rel_with_dir)
    for icls, prob, rel_with_dir in tops:
        print('{:2d} {:.2f} {}'.format(icls, prob, rel_with_dir))


def get_dbp_data(sclasses, batch_size):
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
    dataset = load_rc_data(sclasses, rc_file=rc_out, rc_neg_file=rc0_out, neg_ratio=0.2, shuffle=True, exclude_records=exclude)
    # train_data, val_data = dataset, golden  # using golden set as testing set
    train, val = split(dataset, splits=(0.8, 0.2), batch_size=batch_size)  # usual data load
    return train, val


def main():
    from experiments.ontology.symbols import RC_CLASSES_MAP, RC_CLASSES_MAP_ALL, RC_INVERSE_MAP

    # import spacy
    # nlp = spacy.load('en')  # it is imported from other files for now
    # todo: load trained NER
    from experiments.ontology.data import nlp

    random.seed(2)
    batch_size = 1
    epochs = 2
    model_name = 'noner.dr.noaug.v4.4.all.inv'
    sclasses = RC_CLASSES_MAP_ALL
    inverse = RC_INVERSE_MAP

    # Loading encoder
    # encoder = DBPediaEncoder(nlp, sclasses, inverse_relations=inverse)
    encoder = DBPediaEncoderWithEntTypes(nlp, sclasses, inverse_relations=inverse)

    train_data, val_data = get_dbp_data(sclasses, batch_size)

    train_steps = len(train_data) // batch_size
    val_steps = len(val_data) // batch_size
    nb_negs_tr = len([rr for rr in train_data if not rr.relation])
    nb_negs_val = len([rr for rr in val_data if not rr.relation])
    log.info('data: train: {} (negs: {}); val: {} (negs: {})'
             .format(nb_negs_tr, len(train_data), len(val_data), nb_negs_val))

    # Instantiating new net or loading existing
    # net = DBPediaNet(encoder, timesteps=None, batch_size=batch_size)
    # net.compile2()
    model_path = 'dbpedianet_model_{}_full_epochsize{}_epoch{:02d}.h5'.format(model_name, train_steps, 2)
    net = DBPediaNet.from_model_file(encoder, batch_size, model_path=DBPediaNet.relpath('models', model_path))
    model_name += '.i2'

    log.info('classes: {}; model: {}; epochs: {}'.format(encoder.nbclasses, model_name, epochs))
    net._model.summary(line_length=80)

    net.train(cycle(train_data), epochs, train_steps, cycle(val_data), val_steps, model_prefix=model_name)

    test_data = val_data
    prob_threshold = 0.5
    hits, misses = eye_test(net, test_data, prob_threshold=prob_threshold)
    print('rights: {}/{} with prob_threshold={}'.format(len(hits), len(test_data), prob_threshold))
    evals = net.evaluate(cycle(val_data), val_steps)
    print('evaluated: {}'.format(evals))


if __name__ == "__main__":
    from experiments.ontology.data_structs import RelationRecord

    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)
    main()
