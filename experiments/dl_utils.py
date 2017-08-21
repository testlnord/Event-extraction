import logging as log
from collections import defaultdict
from pprint import pprint

from spacy.en import English

from experiments.ner_tagging.ner_net import NERNet
from experiments.ner_tagging.ngram_encoder import LetterNGramEncoder
from experiments.tags import CategoricalTags


def load_nlp(model_path=None, batch_size=16):
    nlp = English()
    ner_net = load_default_ner_net(model_path=model_path, batch_size=batch_size)
    nlp.pipeline.append(ner_net)
    log.info('load_nlp: adding custom entity (iob) tagger to pipeline')
    return nlp


# todo: make ner_net loading consistent (load_weights or load_model)
def load_default_ner_net(model_path=None, batch_size=16):
    raw_tags = ('O', 'I', 'B')
    tags = CategoricalTags(raw_tags)
    encoder = LetterNGramEncoder.from_vocab_file(tags)
    # ner_net = NERNet.from_model_file(encoder=encoder, batch_size=batch_size, model_path=model_path)
    ner_net = NERNet(encoder=encoder, timesteps=100, batch_size=batch_size)
    ner_net.compile_model()
    ner_net.load_weights()
    return ner_net


def print_confusion_matrix(y_true, y_pred, labels, max_print_width=12):
    """
    Rows are predictions, columns are true labels.
    i.e. when labels are [ant, cat]: if cat(row) * ant(column) = 2
    then 2 times we predicted by mistake that a cat is an ant.
    It is assumed that the negative class comes last.
    :param y_true:
    :param y_pred:
    :param labels: entries of y_true & y_pred
    :param max_print_width:
    :return: confusion matrix (numpy.ndarray)
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    total = cm.sum()
    assert total
    accuracy = cm.trace() / total
    cls_results = {}
    for i, label in enumerate(labels):
        # class_weight = cm[:, i].sum() / total  # fraction of this class from total
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        cls_results[label] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
        }

    l1 = max([len(str(x)) for x in cm.flatten()])
    l2 = max(map(len, labels))
    l = min(max(l1, l2), max_print_width)  # format print width
    def formatter(thing): return '{}'.format(thing)[:l].rjust(l)
    _labels = [formatter(x) for x in labels]
    _label_str = ' '.join([' '*(l+2)] + _labels)
    linewidth = len(_label_str) + l + 2
    old_opts = np.get_printoptions()
    np.set_printoptions(linewidth=linewidth, formatter={'int': formatter}, threshold=np.inf)

    # Print confusion matrix
    print(_label_str)
    _rows = str(cm).split('\n')
    for row_label, row in zip(_labels, _rows):
        print(row_label, row)

    # Print metrics for all classes
    print()
    print(classification_report(y_true=y_true, y_pred=y_pred, labels=labels))
    print('accuracy: {}'.format(accuracy))

    # Print metrics for all classes
    # for label in labels:
    #     metrics = cls_results[label]
    #     print('{}: f1:{f1:.3f} prec:{precision:.3f} recall:{recall:.3f} tp:{tp} fp:{fp} fn:{fn}'
    #           .format(formatter(label), **metrics))

    # all_metrics = defaultdict(list)
    # for label, metrics in cls_results.items():
    #     for metric, value in metrics.items():
    #         all_metrics[metric].append(value)
    # macro = {}
    # for metric, vals in all_metrics.items():
    #     macro[metric] = sum(vals) / len(vals)
    # print('macro measures:')
    # pprint(macro)

    np.set_printoptions(**old_opts)  # restore numpy printoptions
    return accuracy, cls_results
