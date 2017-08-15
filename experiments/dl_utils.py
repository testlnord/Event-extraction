import logging as log
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


def print_confusion_matrix(y_true, y_pred, labels, max_print_width=12, linewidth=240):
    """
    Rows are predictions, columns are true labels.
    i.e. when labels are [ant, cat]: if cat(row) * ant(column) = 2
    then 2 times we predicted by mistake that a cat is an ant.
    :param y_true:
    :param y_pred:
    :param labels: entries of y_true & y_pred
    :param max_print_width:
    :param linewidth:
    :return: confusion matrix (numpy.ndarray)
    """
    from numpy import set_printoptions
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    fp = cm[:-1, -1].sum()  # negatives, classified as positives
    fn = cm[-1, :-1].sum()  # positives, classified as negatives
    tp = sum([cm[i, i] for i in range(len(cm)-1)])
    tn = cm[-1, -1]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    l1 = max([len(str(x)) for x in cm.flatten()])
    l2 = max(map(len, labels))
    l = min(max(l1, l2), max_print_width)  # format print width
    def formatter(thing): return '{}'.format(thing)[:l].rjust(l)
    set_printoptions(linewidth=linewidth, formatter={'int': formatter})
    _labels = [formatter(x) for x in labels]

    _rows = str(cm).split('\n')
    print(' '.join([' '*(l+2)] + _labels))
    for row_label, row in zip(_labels, _rows):
        print(row_label, row)
    print('tn: {}; tp: {}; fp: {}; fn: {}'.format(tn, tp, fp, fn))
    print('precision: {}; recall: {}; f1: {}'.format(precision, recall, f1))
    print()