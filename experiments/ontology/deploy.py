import logging as log
import os
from itertools import combinations

from experiments.nlp_utils import sentences_ents
from experiments.ontology.config import config, load_nlp
from experiments.ontology.symbols import RC_CLASSES_MAP_ALL, RC_CLASSES_MAP_ALL2, RC_CLASSES_MAP_ALL3, RC_INVERSE_MAP, ENT_CLASSES
from experiments.ontology.linker import NERTypeResolver, NERLinker
from experiments.ontology.ont_encoder import DBPediaEncoderEmbed, DBPediaEncoder, DBPediaEncoderBranched
from experiments.ontology.dbpedia_net import DBPediaNet


def get_relation_candidates(doc):
    for sent, sent_ents in sentences_ents(doc):
        # NB: multiple entries of the same entities could produce almost identical pairs
        yield from combinations(sent_ents, 2)


class RelationRanger:
    def __init__(self, model, topn=1, prob_threshold=0.3,
                 min_ents_dist=0):
        self.model = model
        self.topn = topn
        self.prob_threshold = prob_threshold
        self.min_ents_dist = min_ents_dist

        self._allowed_types = ENT_CLASSES  # todo: move to constructor args

    def get_relations(self, doc):
        # filter pairs only with needed ner types (ner-type-resolver?)
        #   there could be some already present in the ontology candidates...
        clss = self._allowed_types
        rel_cands = []
        for s, o in get_relation_candidates(doc):
            if s.label_ in clss and o.label_ in clss:
                ents_dist = min(abs(o.start - s.end), abs(s.start - o.end))
                if ents_dist >= self.min_ents_dist:
                    rel_cands.append((s, o))

        if rel_cands:
            # The format is List[Tuple[subject_span, object_span]]
            rel_preds = self.model.predict(rel_cands, topn=self.topn)
            for rc, preds in zip(rel_cands, rel_preds):
                final_preds = []
                for i, prob, rcls in preds:
                    if prob >= self.prob_threshold:
                        final_preds.append((rcls, prob))
                yield rc, final_preds


def test_pipeline():
    models_dir = config['models']['dir']

    # Load models: either default or trained ones
    # nlp_model_name = 'models.v5.4.i{}.epoch{}'.format(1, 8)
    # nlp_model_name = 'models.v5.4.i{}.epoch{}'.format(5, 2)
    nlp_model_name = 'models.cls.v7.1.i{}.epoch{}'.format(5, 4)
    nlp = load_nlp(nlp_model_name)

    ntr = NERTypeResolver()
    linker = NERLinker(ner_type_resolver=ntr)
    linker.load(models_dir)
    nlp.pipeline.append(linker)

    sclasses = RC_CLASSES_MAP_ALL
    inverse = RC_INVERSE_MAP
    encoder = DBPediaEncoderEmbed(nlp, sclasses, inverse, ner_type_resolver=ntr,
                                  min_entities_dist=2, expand_context=3)
    # Load one of the models trained with that encoder
    # model_name = 'cls.v7.1.c4.our_nlp.inv'
    model_name = 'cls.v7.1.c4l3.spacy.inv'  # epoch: 4
    # model_name = 'cls.v7.1.c4.l3.spacy.rc2.inv'  # epoch: 0
    # model_name = 'cls.full_sents.v7.2.c4.l3.spacy.rc3.inv'  # epoch: 1
    # model_name = 'cls.v7.2.c4.l3.spacy.rc3.inv'  # epoch: 1
    model_name = 'dbpedianet_model_{}_full_epoch{:02d}.h5'.format(model_name, 4)

    model_path = os.path.join(models_dir, model_name)
    net = DBPediaNet.from_model_file(encoder, batch_size=1, model_path=model_path)

    rranger = RelationRanger(net, topn=3, prob_threshold=0.25, min_ents_dist=2)
    # titles = ['JetBrains', 'Microsoft_Windows']
    titles = ['Blender_(software)', 'JetBrains']
    test_articles(nlp, linker, rranger, titles)


def test_articles(nlp, linker, rranger, titles):
    from experiments.ontology.sub_ont import dbr, get_article

    _enc = rranger.model._encoder  # for getting and outputting sdp

    for i, title in enumerate(titles):
        print('\n')
        print(i, title)
        text = get_article(dbr[title])['text']

        # This is how texts should be processed:
        doc = nlp(text)  # NERLinker does its' job here
        for i, ((s, o), preds) in enumerate(rranger.get_relations(doc)):
            sdp = _enc._encode_sdp(s, o)
            print()
            print(s.sent.text.strip())
            print(sdp)
            print('{} <({}) {}: {}> ----- <({}) {}: {}>'
                  .format(i, s.label_, s, linker.get(s), o.label_, o, linker.get(o)))
            for rcls, prob in preds:
                print('{:.2f} {}'.format(prob, rcls))


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.DEBUG)

    # todo: make import unnecessary
    # This import is needed for linker.load
    from experiments.ontology.linker import TrieEntry

    test_pipeline()
