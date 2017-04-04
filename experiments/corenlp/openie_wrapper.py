# from corenlp_pywrap import pywrap
from pycorenlp import StanfordCoreNLP
from pprint import pprint

from spacy.en import English
from event import Event
from experiments.extraction import *

from collections import defaultdict
# import operator as op
# from functools import reduce
# from intervals import IntInterval

class StanfordOpenIE:
    # oie_pipeline = "tokenize,ssplit,pos,depparse,natlog,openie"
    output_format = "json"

    def __init__(self, port=9000):
        # default CoreNLP port
        host = "localhost"
        url = "http://{}:{}".format(host, port)
        self.cn = StanfordCoreNLP(url)

    def get_relations(self, sentences):
        properties = {
            # 'annotators': self.oie_pipeline,
            'annotators': 'openie',
            'outputFormat': self.output_format,
            'openie.triple.strict': 'false',
            'openie.resolve_coref': 'false',
        }
        for i, sent in enumerate(sentences):
            output = self.cn.annotate(str(sent), properties=properties)
            output = output['sentences'][0]
            rels = output['openie']
            tokens = output['tokens']

            # yield [self.relation2event(sent, rel, tokens) for rel in rels]
            # yield list(self._extract(sent, rels, tokens))

            ### just logging-checking
            print('NEXT', i)
            print(sent)
            # pprint(tokens)
            for e in self._extract(sent, rels, tokens):
                print()
                print('S0: ', e.str2)
                print('S1: ', e)
                for relation in rels:
                    # e = self.relation2event(sent, relation, tokens)
                    _subj = relation['subject']
                    _rel = relation['relation']
                    _obj = relation['object']
                    pprint(relation)
                    print(' C: ', '; '.join([_subj, _rel, _obj]))
                    # print(' S: ', e)

    def _extract(self, span, relations, tokens):
        all_rels = defaultdict(lambda: set())

        # todo: find min and max simultaneously in one pass in first loop?
        for relation in relations:
            subj_span = relation['subjectSpan']
            rel_span = relation['relationSpan']
            obj_span = relation['objectSpan']
            # Find correspondence between CoreNLP tokens and Spacy tokens using information about char offsets
            subj_span = self._corenlp_span_to_spacy_span(subj_span, span, tokens)
            rel_span = self._corenlp_span_to_spacy_span(rel_span, span, tokens)
            obj_span = self._corenlp_span_to_spacy_span(obj_span, span, tokens)

            # todo: 1hash by whole intervals or their roots? -- WHOLE INTERVALS
            all_rels[(subj_span, rel_span)].add(obj_span)

        def spanlen(s): return s[1] - s[0]
        for (subj_span, rel_span), related_rels in all_rels.items():
            min_subtree = min(related_rels, key=spanlen)
            max_subtree = max(related_rels, key=spanlen)

            e = Extraction3(span, subj_span, rel_span, min_subtree, max_subtree)
            yield e

    def _corenlp_span_to_spacy_span(self, corenlp_span, spacy_span, tokens):
        cai, cbi = corenlp_span
        ca = tokens[cai]['characterOffsetBegin']
        cb = tokens[cbi]['characterOffsetEnd']
        splen = len(spacy_span)
        slices = list(zip(range(splen), list(range(1, splen)) + [-1]))

        spacy_token_first = 0
        for spacy_token_first, b in slices:
            stoken = spacy_span[spacy_token_first:b]
            if stoken.start_char >= ca:
                break
        spacy_token_last = 0
        for spacy_token_last, b in slices:
            stoken = spacy_span[spacy_token_last:b]
            if stoken.end_char >= cb:
                break
        offset = (spacy_token_first, spacy_token_last)
        return offset



    ############################## temporal
    def full_relation2extraction(self, span, relation, tokens):
        _subj = relation['subject']
        _rel = relation['relation']
        _obj = relation['object']
        _subj_span = relation['subjectSpan']
        _rel_span = relation['relationSpan']
        _obj_span = relation['objectSpan']
        _subj = self._extract_part(span, _subj, _subj_span, tokens)
        _obj = self._extract_part(span, _obj, _obj_span, tokens)
        _rel = self._extract_part(span, _rel, _rel_span, tokens)
        return Extraction2(span, subject_indices=_subj, relation_indices=_rel, object_indices=_obj)

    def _extract_part(self, spacy_span, part, corenlp_span, tokens):
        cai, cbi = corenlp_span
        ca = tokens[cai]['characterOffsetBegin']
        cb = tokens[cbi]['characterOffsetEnd']

        result_token_offsets = []
        # todo adjust all indices according to CA and CB - CHECK IT
        stext = spacy_span.text[ca:cb]
        stokens_starts = [stoken.idx - spacy_span.start_char for stoken in spacy_span]
        stokens_ends = [stoken.idx + len(stoken) - spacy_span.start_char for stoken in spacy_span]
        itoken = 0
        i = 0
        j = 0
        while i < cb and j < len(part):
            while i < cb and stext[i] != part[j]:
                i += 1
            while stokens_ends[itoken] < i:
                itoken += 1

            #while i < cb and j < len(part) and stext[i] == part[j]:
            while i < cb and stext[i] == part[j]:
                i += 1
                j += 1
            while itoken < len(stokens_ends) and stokens_ends[itoken] <= i:
                result_token_offsets.append(itoken)
                itoken += 1
        return result_token_offsets

    def relation2event(self, span, relation, tokens):
        # todo: really test char offset extraction vs. token offset extraction
        _subj_span = relation['subjectSpan']
        _rel_span = relation['relationSpan']
        _obj_span = relation['objectSpan']

        # Find correspondence between CoreNLP tokens and Spacy tokens using information about char offsets
        _subj_span = self._corenlp_span_to_spacy_span(_subj_span, span, tokens)
        _rel_span = self._corenlp_span_to_spacy_span(_rel_span, span, tokens)
        _obj_span = self._corenlp_span_to_spacy_span(_obj_span, span, tokens)

        return Extraction(span, _subj_span, _rel_span, _obj_span)



if __name__ == "__main__":
    cn = StanfordOpenIE()
    nlp = English()
    with open("/home/hades/projects/Event-extraction/experiments/data/samples.txt") as f:
        sents = [nlp(sent)[:] for sent in f.readlines()]
        # sents = list(f.readlines())
    cn.get_relations(sents)
