import logging as log
import os
import csv
import re
import pickle

import numpy as np
from rdflib import URIRef
from experiments.ontology.sub_ont import get_contexts, get_article
from experiments.ontology.sub_ont import raw, gfall, gf
from experiments.ontology.ont_encoder import filter_context


# Read list of valid properties from the file
props_dir = '/home/user/datasets/dbpedia/qs/props/'
props = props_dir + 'all_props_nonlit.csv'
with open(props, 'r', newline='') as f:
    prop_reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    valid_props = [URIRef(prop) for prop, n in prop_reader]

def validate(s, r, o):
    """Check if triple is one we want in the dataset."""
    return r in valid_props


contexts_dir = '/home/user/datasets/dbpedia/contexts/'
delimiter = ' '
quotechar = '|'
quoting = csv.QUOTE_NONNUMERIC


def make_dataset(triples, output_path, mode='w'):
    log_total = 0
    header = ['s', 'r', 'o', 's_start', 's_end', 'o_start', 'o_end', 'context', 'context_start', 'context_end', 'article_id']
    with open(output_path, mode, newline='') as f:
        writer = csv.writer(f, delimiter=delimiter, quotechar=quotechar, quoting=quoting)  # todo:adjust
        if mode == 'w': writer.writerow(header)
        for i, triple in enumerate(triples, 1):
            if validate(*triple):
                log.info('make_dataset: processing triple #{}: {}'.format(i, [str(t) for t in triple]))
                for j, (ctx, s0, s1, o0, o1, art) in enumerate(get_contexts(*triple), 1):
                    # write both the text and its' source
                    log_total += 1
                    log.info('make_dataset: contex #{} (total: {})'.format(j, log_total))
                    writer.writerow(list(triple) + [s0, s1, o0, o1, ctx.text.strip(), ctx.start_char, ctx.end_char] + [int(art['id'])])


class RelationRecord:
    def __init__(self, s, r, o, s0, s1, o0, o1, ctext, cstart, cend, artid):
        # self.s = self.subject = URIRef(s)
        # self.r = self.relation = URIRef(r)
        # self.o = self.object = URIRef(o)
        self.s = self.subject = s
        self.r = self.relation = r
        self.o = self.object = o
        self.s0 = self.s_start = s0
        self.s1 = self.s_end = s1
        self.o0 = self.o_start = o0
        self.o1 = self.o_end = o1
        self.context = ctext
        self.cstart = cstart
        self.cend = cend
        self.article_id = artid

    def cut_context(self, begin, end):
        self.context = self.context[begin:end]
        self.cstart = self.cstart + begin
        self.cend = self.cstart + end
        self.s0 = self.s_start = max(self.cstart, self.s0)
        self.s1 = self.s_end = min(self.s1, self.cend)
        self.o0 = self.o_start = max(self.cstart, self.o0)
        self.o1 = self.o_end = min(self.o1, self.cend)

    @property
    def triple(self): return (self.subject, self.relation, self.object)

    @property
    def s_startr(self): return self.s_start - self.cstart  # offsets in dataset are relative to the whole document, not to the sentence

    @property
    def s_endr(self): return self.s_end - self.cstart

    @property
    def o_startr(self): return self.o_start - self.cstart

    @property
    def o_endr(self): return self.o_end - self.cstart

    @property
    def s_spanr(self): return (self.s_startr, self.s_endr)

    @property
    def o_spanr(self): return (self.o_startr, self.o_endr)


class EntityRecord:
    def __init__(self, crecord, start, end, uri):
        """

        :param uri: original uri
        :param start: char offset of the start in crecord.context (including start)
        :param end: char offset of the end in crecord.context (not including end)
        :param crecord: source of entity: ContextRecord
        """
        self.crecord = crecord
        self.start = start
        self.end = end
        self.uri = uri

    @property
    def text(self):
        return self.crecord.context[slice(*self.span)]

    @property
    def span(self):
        return self.start, self.end

    @property
    def spang(self):
        s = self.crecord.start
        return s+self.start, s+self.end

    def cut_context(self, begin, end):
        self.start = max(0, self.start - begin)
        self.end = min(self.end, end)


class ContextRecord:
    @classmethod
    def from_span(cls, span, artid, ents=None):
        return cls(span.text, span.start_char, span.end_char, artid, ents)

    def __init__(self, ctext, cstart, cend, artid, ents=None):
        self.context = ctext
        self.start = cstart
        self.end = cend
        self.article_id = artid
        self.ents = [] if ents is None else ents

    def cut_context(self, begin, end):
        self.context = self.context[begin:end]
        self.start = self.start + begin
        self.end = self.start + end
        for e in self.ents:
            e.cut_context(begin, end)

    @property
    def span(self):
        return self.start, self.end


def read_dataset(path):
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar, quoting=quoting)
        header = next(reader)
        log.info('read_dataset: header: {}'.format(header))
        # for s, r, o, s0, s1, o0, o1, ctext, cstart, cend, artid in reader:
        for data in reader:
            yield RelationRecord(*data)


### Auxiliary functions ###


props_dir='/home/user/datasets/dbpedia/qs/classes/'


def load_superclass_mapping(filename=props_dir+'prop_classes.csv'):
    """Mapping between relations and superclasses"""
    classes = {}
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC, delimiter=' ')
        for row in reader:
            icls, cls, rel = row[:3]
            if int(icls) >= 0:
                classes[rel] = cls
    return classes


def load_inverse_mapping(filename=props_dir+'prop_inverse.csv'):
    """Mapping between superclasses and their inverse superclasses"""
    inverse = {}
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC, delimiter=' ')
        for icls, cls, iinv, inv in reader:
            inverse[cls] = inv
            inverse[inv] = cls
    return inverse


def load_all_data(classes, data_dir=contexts_dir, shuffle=True):
    """Load ContextRecords with classes from @classes from all files in the directory @data_dir and shuffle it."""
    dataset = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            for crecord in read_dataset(filepath):
                if crecord.relation in classes:
                    filtered = filter_context(crecord)
                    if filtered is not None:
                        dataset.append(filtered)
    if shuffle: np.random.shuffle(dataset)
    return dataset







from collections import defaultdict
from fuzzywuzzy import fuzz
from experiments.ontology.sub_ont import nlp, get_label



# Complexity: O(O(metric)*len(doc)*len(candidates))
def fuzzfind_plain1(doc, *candidates, metric=fuzz.ratio, metric_threshold=82):
    # Merge noun_chunks and entities to iterate over them along with the simple tokens
    for e in doc.noun_chunks: e.merge()
    for e in doc.ents: e.merge()
    for k, sent in enumerate(doc.sents):
        found = defaultdict(list)
        for i, t in enumerate(sent):
            # Choosing 'better matching' entity
            cent, measured = max([(c, metric(t, c)) for c in candidates], key=lambda tpl: tpl[1])
            if measured >= metric_threshold:
                ii = t.idx
                found[cent].append((ii, ii+len(t)))
        yield sent, found


def get_contexts0(articles, *ents_uris):
    """
    Search for occurences of ents in the articles about ents.
    :param articles: mapping type, containing fields 'text' and 'id'
    :param ents_uris: uris of the ents
    :yield: ContextRecord
    """
    raw_ents = {get_label(ent_uri): ent_uri for ent_uri in ents_uris}
    for i, article in enumerate(articles, 1):
        log.info('get_contexts: article #{} with title:"{}"'.format(i, article['title']))
        doc = nlp(article['text'])
        for sent, ents_dict in fuzzfind_plain1(doc, *raw_ents.keys()):
            crecord = ContextRecord.from_span(sent, artid=article['id'])
            crecord.ents = [EntityRecord(crecord, a, b, uri=raw_ents[raw_ent]) for raw_ent, (a, b) in ents_dict.items()]
            yield crecord



def make_ner_dataset(output_file, visited=set()):
    with open(output_file, 'wb') as f:
        for subject in gf.subjects():
            objects = set(gf.objects(subject=subject))
            # todo: What articles to add to art_ids
            articles = filter(None, map(get_article, objects))
            articles = list(filter(lambda a: a['id'] not in visited, articles))

            subj_article = get_article(subject)
            objects.add(subject)
            articles.append(subj_article)
            for crecord in get_contexts0(articles, *objects):
                # todo: save
                pickle.dump(crecord, f)

            if subj_article is not None:
                # for crecord in get_contexts0([subj_article], objects):
                visited.add(subj_article['id'])

### Tests ###


def test(triples):
    for triple in triples:
        ctxs = list(get_contexts(*triple))
        print(len(ctxs), triple, '\n')
        for i, ctx_data in enumerate(ctxs):
            print('_' * 40, i)
            print(ctx_data[0])


def test_count_data(valid_props=valid_props):
    # Count how much data did we get
    total = total_extracted = total_filtered = 0
    for prop in valid_props:
        print('\n\nPROPERTY:', prop)
        filename = contexts_dir+'test4_{}.csv'.format(raw(prop))
        data = list(read_dataset(filename))
        triples = list(gfall.triples((None, prop, None)))
        filtered = 0
        for i, crecord in enumerate(data, 1):
            # s, r, o, s0, s1, o0, o1, ctext, cstart, cend, artid = crecord
            if re.search('\n+', crecord.context.strip()):
                print(i, crecord.triple)
                print(crecord.context)
                fct = filter_context(crecord)
                filtered += bool(fct is None)
                print('<{}>'.format(fct.context if fct is not None else None))
        total_filtered += filtered
        # Count
        nb_all = len(triples)
        nb_extracted = len(data)
        print('{}: {}/{} ~{:.2f}'.format(prop, nb_extracted, nb_all, nb_extracted/nb_all))
        total += nb_all
        total_extracted += nb_extracted
    tt = total_extracted - total_filtered
    print('filtered/extracted: {}/{} ~{:.2f}'.format(total_filtered, total_extracted, total_filtered/total_extracted))
    print('total: {}/{} ~{:.2f}'.format(tt, total, tt/total))


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    # jbtriples = list(gf.triples((dbr.JetBrains, dbo.product, None)))
    # mtriples = list(gf.triples((dbr.Microsoft, dbo.product, None)))
    # test(jbtriples)
    # exit()

    # test_count_data()


    # Make NER dataset
    visited = set()
    output_file = '/home/user/datasets/dbpedia/ner/' + 'crecords.pck'
    make_ner_dataset(output_file, visited)
    # try:
    #     pass
    # except Exception as e:
    #     print(e)
    #     print(visited)  # dump if something bad happens

    # Make dataset
    # for prop in valid_props:
    #     triples = gfall.triples((None, prop, None))
    #     filename = contexts_dir+'test0_{}.csv'.format(raw(prop))
    #     make_dataset(triples, filename)

    # prop = dbo.product
    # triples = gfall.triples((None, prop, None))
    # make_dataset(triples, contexts_dir+'test3_{}.csv'.format(raw(prop)))
