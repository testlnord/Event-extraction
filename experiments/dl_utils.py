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
