import logging as log
import os
from itertools import permutations, combinations

from cytoolz import groupby
from intervaltree import IntervalTree
import spacy

from experiments.nlp_utils import sentences_ents
from experiments.ontology.config import config
from experiments.ontology.symbols import RC_CLASSES_MAP, RC_CLASSES_MAP_ALL, RC_INVERSE_MAP, ENT_CLASSES
from experiments.ontology.linker import NERTypeResolver, NERLinker
from experiments.ontology.ont_encoder import *
from experiments.ontology.dbpedia_net import DBPediaNet
# from experiments.ontology.


def get_relation_candidates(doc):
    for sent, sent_ents in sentences_ents(doc):
        # NB: multiple entries of the same entities could produce almost identical pairs
        yield from combinations(sent_ents, 2)


# todo: output format
class RelationRanger:
    def __init__(self, model, topn=1, prob_threshold=0.3):
        self.model = model
        self.topn = topn
        self.prob_threshold = prob_threshold

        self._allowed_types = ENT_CLASSES  # todo: remove dependency?

    def get_relations(self, doc):
        # filter pairs only with needed ner types (ner-type-resolver?)
        #   there could be some already present in the ontology candidates...
        clss = self._allowed_types
        rel_cands = [(s, o) for s, o in get_relation_candidates(doc) if s.label_ in clss and o.label_ in clss]

        if rel_cands:
            # The format is List[Tuple[subject_span, object_span]]
            rel_preds = self.model.predict(rel_cands, topn=self.topn)
            for rc, preds in zip(rel_cands, rel_preds):
                final_preds = []
                for i, prob, rcls in preds:
                    if prob >= self.prob_threshold:
                        final_preds.append((rcls, prob))
                yield rc, final_preds


def main1():
    _models_ = config['models']
    models_dir = os.path.join(config['base_dir'], _models_['dir'])

    # model_dir = '/home/user/projects/Event-extraction/experiments/ontology'
    # model_name = 'models.v6.1.i{}.epoch{}'.format(10, 4)
    # nlp = spacy.load('en', path=os.path.join(model_dir, model_name))
    nlp = spacy.load(**_models_['nlp'])

    ntr = NERTypeResolver()
    linker = NERLinker(ner_type_resolver=ntr)
    linker.load(models_dir)
    nlp.pipeline.append(linker)

    sclasses = RC_CLASSES_MAP_ALL
    inverse = RC_INVERSE_MAP
    encoder = DBPediaEncoderEmbed(nlp, sclasses, inverse_relations=inverse, ner_type_resolver=ntr)
    model_name = 'nocls.v6.3.c4.spacy.inv'
    model_name = 'dbpedianet_model_{}_full_epoch{:02d}.h5'.format(model_name, 4)
    model_path = os.path.join(models_dir, model_name)
    net = DBPediaNet.from_model_file(encoder, batch_size=1, model_path=model_path)

    rranger = RelationRanger(net, topn=3, prob_threshold=0.25)
    test_articles(nlp, linker, rranger)


def main2():
    _models_ = config['models']
    models_dir = os.path.join(config['base_dir'], _models_['dir'])

    model_dir = '/home/user/projects/Event-extraction/experiments/ontology'
    # model_name = 'models.v5.4.i{}.epoch{}'.format(1, 8)
    model_name = 'models.v5.4.i{}.epoch{}'.format(5, 2)
    nlp = spacy.load('en', path=os.path.join(model_dir, model_name))

    ntr = NERTypeResolver()
    linker = NERLinker(ner_type_resolver=ntr)
    linker.load(models_dir)
    nlp.pipeline.append(linker)

    sclasses = RC_CLASSES_MAP_ALL
    inverse = RC_INVERSE_MAP
    encoder = DBPediaEncoder(nlp, sclasses, inverse_relations=inverse, ner_type_resolver=ntr)
    # model_name = 'noner.dr.noaug.v5.1.c3.all.inv'
    model_name = 'noner.dr.noaug.v6.3.c3.all.inv'
    model_name = 'dbpedianet_model_{}_full_epoch{:02d}.h5'.format(model_name, 2)
    model_path = os.path.join(models_dir, model_name)
    net = DBPediaNet.from_model_file(encoder, batch_size=1, model_path=model_path)

    rranger = RelationRanger(net, topn=3, prob_threshold=0.25)
    test_articles(nlp, linker, rranger)


def test_articles(nlp, linker, rranger):
    from experiments.ontology.sub_ont import dbr, get_article

    # titles = [dbr.JetBrains, dbr.Microsoft_Windows]
    titles = [dbr['Blender_(software)'], dbr.FreeMind]
    for i, title in enumerate(titles):
        print('\n')
        print(i, str(title))
        text = get_article(title)['text']

        # This is how texts should be processed:
        doc = nlp(text)  # NERLinker does its' job here
        for i, ((s, o), preds) in enumerate(rranger.get_relations(doc)):
            print()
            print(s.sent.text.strip())
            print('{} <({}) {}: {}> ----- <({}) {}: {}>'
                  .format(i, s.label_, s, linker.get(s), o.label_, o, linker.get(o)))
            for rcls, prob in preds:
                print('{:.2f} {}'.format(prob, rcls))


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.DEBUG)

    from experiments.ontology.linker import TrieEntry  # for linker.load --- todo: make import unnecessary

    main1()
