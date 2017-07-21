import logging as log
from time import sleep
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


class except_safe:
    def __init__(self, *exceptions, tries=10):
        self.exceptions = exceptions
        self.tries = tries

    def __call__(self, f):
        def safe_f(*args):
            for i in range(1, self.tries+1):
                try:
                    return f(*args)
                except self.exceptions as e:
                    if i == self.tries:
                        raise e
                    log.warning('except_safe decorator: try #{} (next delay {}s): {}: {}'.format(i, i-1, f.__name__, e))
                    sleep(i-1)
        return safe_f


@except_safe(Exception, tries=4)
def test_except_safe(n_exceptions, buf=[]):
    l = len(buf)
    print(test_except_safe.__name__, l)
    if l < n_exceptions:
        buf.append(l)
        raise Exception('dummy exception')


if __name__ == "__main__":
    test_except_safe(3)  # no exception is thrown
    test_except_safe(4)  # exception is thrown on 4th execution

