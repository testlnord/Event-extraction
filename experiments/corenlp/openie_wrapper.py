import logging as log
from collections import defaultdict
from pprint import pprint

from pycorenlp import StanfordCoreNLP
from spacy.en import English

from experiments.extraction.extraction import Extraction


class StanfordOpenIE:
    version = 1  # todo: use CoreNLP.OpenIE version?
    # oie_pipeline = "tokenize,ssplit,pos,lemma,depparse,natlog,openie"
    oie_pipeline = "tokenize,ssplit,pos,lemma,ner,depparse,natlog,mention,coref,openie"
    properties = {
        'annotators': oie_pipeline,
        # 'annotators': 'openie',
        'outputFormat': 'json',
        'openie.triple.strict': 'false',
        'openie.resolve_coref': 'true',
    }

    def __init__(self, port=9000):
        # default CoreNLP port
        host = "localhost"
        url = "http://{}:{}".format(host, port)
        if self._reachable(host, port):
            self.cn = StanfordCoreNLP(url)
            log.info('StanfordOpenIE: Connected to Stanford CoreNLP server')
        else:
            raise ConnectionError('Stanford CoreNLP server is not running on ({}, {})!'.format(host, port))

    # todo: move to utils
    def _reachable(self, host, port):
        from socket import socket, error
        s = socket()
        try:
            s.connect((host, port))
            return True
        except error:
            return False

    def process(self, sentence):
        output = self.cn.annotate(str(sentence), properties=self.properties)
        output = output['sentences'][0]
        rels = output['openie']
        tokens = output['tokens']

        for e in self._extract(sentence, rels, tokens):
            yield e

    def get_relations(self, sentences):
        for i, sent in enumerate(sentences):
            log.info('Sending data to CoreNLP server...')
            output = self.cn.annotate(str(sent), properties=self.properties)
            log.info('Received data from CoreNLP server')
            output = output['sentences'][0]
            rels = output['openie']
            tokens = output['tokens']

            # yield [self.relation2event(sent, rel, tokens) for rel in rels]
            # yield list(self._extract(sent, rels, tokens))

            ### just logging-checking
            print('NEXT', i)
            print(sent)
            # pprint(tokens)
            if False:
                for relation in rels:
                    # e = self.relation2event(sent, relation, tokens)
                    _subj = relation['subject']
                    _rel = relation['relation']
                    _obj = relation['object']
                    pprint(relation)
                    print(' C: ', '; '.join([_subj, _rel, _obj]))
                    # print(' S: ', e)

            for e in self._extract(sent, rels, tokens):
                print()
                old = e.object_max_span
                # ne_filter.match(e)
                if e.object_max_span != old:
                    print('S0: ', e.str2)
                    print('S1: ', e._substring(old))
                    # print('SF: ', e.object_max)

    def _extract(self, span, relations, tokens):
        all_rels = defaultdict(lambda: set())

        for relation in relations:
            subj_span = tuple(relation['subjectSpan'])
            rel_span = tuple(relation['relationSpan'])
            obj_span = tuple(relation['objectSpan'])

            # subj_span = self._corenlp_span_to_spacy_span(subj_span, span, tokens)
            # rel_span = self._corenlp_span_to_spacy_span(rel_span, span, tokens)
            # obj_span = self._corenlp_span_to_spacy_span(obj_span, span, tokens)

            all_rels[(subj_span, rel_span)].add(obj_span)

        def spanlen(s): return s[1] - s[0]
        for (subj_span, rel_span), related_rels in all_rels.items():
            min_subtree = min(related_rels, key=spanlen)
            max_subtree = max(related_rels, key=spanlen)

            # Remove intersection between spans
            min_subtree = self._disjoin(subj_span, min_subtree)
            min_subtree = self._disjoin(rel_span, min_subtree)
            max_subtree = self._disjoin(subj_span, max_subtree)
            max_subtree = self._disjoin(rel_span, max_subtree)

            yield Extraction(span, subj_span, rel_span, min_subtree, max_subtree)

    def _disjoin(self, main, second):
        """Remove intersection (if it exists) between main and second from the second"""
        a0, b0 = main
        a1, b1 = second
        if a0 <= a1 < b0:
            a1 = b0
        elif a0 < b1 <= b0:
            b1 = a0
        return a1, b1

    # todo: find mistake
    def _corenlp_span_to_spacy_span(self, corenlp_span, spacy_span, tokens):
        """
        Translate CoreNLP's token indices to spacy's token indices using information about char offsets of tokens in original text.
        :param corenlp_span: 
        :param spacy_span: 
        :param tokens: CoreNLP's tokens list with information about char offsets
        :return: 
        """
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

    # old, not tested
    def _extract_part(self, spacy_span, part, corenlp_span, tokens):
        # extract tokens by char offsets
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


from experiments.extraction.extraction_filters import Extractor
if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.INFO)

    cn = StanfordOpenIE()
    nlp = English()
    # nlp = spacy.load('en')
    extractor = Extractor(nlp)
    with open("/home/hades/projects/Event-extraction/experiments/data/samples.txt") as f:
        sents = [nlp(sent) for sent in f.readlines()]
        for sent in sents:
            for e in cn.process(sent):
                print()
                print('SS: ', e.span)
                print('S0: ', e.str2)
                print('S1: ', e)
                ents, rels, attrs = extractor.process(e)
                print('ents', ents)
                print('rels', rels)
                print('attrs', attrs)
        # sents = list(f.readlines())
    # cn.get_relations(sents)
