import logging as log
from rdflib.graph import Dataset, Graph, ReadOnlyGraphAggregate
from rdflib.namespace import RDF, RDFS, OWL, FOAF, Namespace, NamespaceManager, URIRef
from rdflib.store import Store
from rdflib.plugins.stores.sparqlstore import SPARQLStore, SPARQLUpdateStore
from SPARQLWrapper import DIGEST, URLENCODED, POSTDIRECTLY, POST, RDFXML, TURTLE
from rdflib.plugins.stores.sparqlstore import SPARQLWrapper
import json
import csv
from collections import defaultdict

log.basicConfig(format='%(levelname)s:%(message)s', level=log.INFO)
dbp_dir = '/home/user/datasets/dbpedia/'
basedir = '/home/user/datasets/dbpedia/z2016_10/'
dir_articles = '/home/user/datasets/dbpedia/articles/'

update_endpoint = 'http://localhost:8890/sparql-auth'
endpoint = 'http://localhost:8890/sparql'  # no update (insert, create graph, whatever...) access
store = SPARQLUpdateStore(endpoint, update_endpoint, autocommit=True)  # need to call store.commit explicitly todo: there's some trouble in sparqlstore's source with that autocommit logic
store.setHTTPAuth(DIGEST)
store.setCredentials(user='dba', passwd='admin')
ds = Dataset(store, default_union=False)  # todo: default_union ?

dbo = Namespace('http://dbpedia.org/ontology/')
dbr = Namespace('http://dbpedia.org/resource/')
iri_dbo = 'http://dbpedia.org/ontology'
iri_dbpedia = 'http://dbpedia.org'
iri_labels = 'http://dbpedia.org/labels'
iri_field = 'field'
iri_more = 'field:more'

# NB: for existing graphs use 'ds.get_context(iri)', for new graphs use 'ds.graph(iri)'
# errors or silent ignoring of actions (on non-existing graphs) otherwise should be expected...
# remove_graph also requires already existing graph
gdb = ds.get_context(iri_dbpedia)
gdbo = ds.get_context(iri_dbo)
glabels = ds.get_context(iri_labels)
glo = ReadOnlyGraphAggregate([gdbo, glabels])
gall = ReadOnlyGraphAggregate([gdb, gdbo])
gf = ds.get_context(iri_field)
gmore = ds.get_context(iri_more)
gfall = ReadOnlyGraphAggregate([gf, gmore])


# Some simple tests
# gtest = ds.get_context('gtest')
# gtest.update('INSERT DATA {<s1> <r1> <o1> }')
# gtest.add((URIRef('s1'), URIRef('r1'), URIRef('o1')))  # analogous
# print(len(gtest.query('SELECT * WHERE {?s ?r ?o}')))
# gtest.update('DELETE DATA {<s1> <r1> <o1> }')
# print(len(gtest.query('SELECT * WHERE {?s ?r ?o}')))

qheaders = [
    '''DEFINE input:inference <http://dbpedia.org/ontology>''',
    '''CONSTRUCT {?s ?r ?o} ''' + 'FROM <{}> FROM <{}> '.format(iri_dbpedia, iri_dbo),
]
_queries = [
    '''WHERE { ?s ?r ?o .
         {?s rdf:type dbo:Organisation .}
         {?o rdf:type dbo:ProgrammingLanguage .} UNION {?o rdf:type dbo:Software .}
         MINUS {?s dbp:format ?o}
       }
    ''',
]
queries = ['\n'.join(qheaders + [q]) for q in _queries]

def add_triples(query):
    """
    Add triples from CONSTRUCT query to subgraph.
    Query can be of any complexity (using FROM and FROM NAMED simultaneously, for instance)
    """
    qres = ds.query(query)  # gen-r of results of the construct query
    for res in qres:
        gf.add(res)  # adding to graph 'gf' in RDF Database (context is the graph 'gf')

def get_article(subject):
    """Fail-safe, when article is not present."""
    try:
        id_uri = next(gdb.objects(subject=subject, predicate=dbo.wikiPageID))
    except StopIteration:
        return None
    # _id_end = id_uri.split('"')[0]
    # _id = id_uri[0:int(_id_end)]
    try:
        with open(dir_articles + id_uri) as f:
            art = json.load(f)
        text = art['text']
        first_par = text.find('\n\n')  # cut the title
        art['text'] = text[first_par+2:]
        return art
        # return art['id'], text[first_par+2:]
    except FileNotFoundError:
        return None

def raw(uri):
    return uri.rsplit('/', 1)[-1]

def get_label(uri):
    t = list(glo.objects(uri, RDFS.label))
    t = raw(uri) if len(t) == 0 else str(t[0])
    return t.split('(')[0].strip(' _')  # remove disambiguations

from fuzzywuzzy import fuzz
from itertools import product
import editdistance
from spacy.en import English
from experiments.utils import load_nlp

nlp = English()
# nlp = load_nlp()
# nlp = load_nlp(batch_size=32)

def make_fuzz_metric(fuzz_ratio=80):
    def fz(t1, t2):
        return fuzz.ratio(t1, t2) >= fuzz_ratio
    return fz

def make_sim_metric(similarity_threshold):
    def sm(t1, t2):
        return t1.similarity(t2) >= similarity_threshold
    return sm

def make_metric(ratio=80, partial_ratio=95):
    def m(x, y):
        fzr = fuzz.ratio(x, y)
        fzpr = fuzz.partial_ratio(x, y)
        return (fzr >= ratio) or (fzpr >= partial_ratio and fzr >= ratio / 2)
    return m

metric = make_fuzz_metric()

def fuzzfind_plain(doc, s, r, o):
    for k, sent in enumerate(doc.sents):
        is_subject = is_object = is_relation = False
        for j, token in enumerate(sent):
            is_subject |= metric(token.text, s)
            is_object |= metric(token.text, o)
            # is_relation |= (token.similarity(nlp(r)) >= sim)
            if is_subject and is_object:
                yield sent
                break

def fuzzfind(doc, s, r, o, fuzz_ratio=85, yield_something=True):
    """Fuzzy search of multiple substrings in a spacy doc; returning a covering span of all of them."""
    s_ents = []
    o_ents = []
    for ent in doc.ents:
        if fuzz.ratio(ent.text, s) >= fuzz_ratio:
            s_ents.append(ent)
        elif fuzz.ratio(ent.text, o) >= fuzz_ratio:
            o_ents.append(ent)
    pairs = list(sorted(product(s_ents, o_ents), key=lambda p: abs(p[0].start - p[1].start)))  # preference for nearer matches
    for s_ent, o_ent in pairs:
        o_sent = o_ent.sent
        s_sent = s_ent.sent
        if s_sent == o_sent:  # choosing matches in one sentence
            yield_something = False
            yield s_sent
    if yield_something and len(pairs) > 0:
        s_sent = pairs[0][0].sent
        o_sent = pairs[0][1].sent
        yield doc[min(s_sent.start, o_sent.start): max(s_sent.end, o_sent.end)]

def get_contexts(s, r, o):
    stext = get_label(s)
    rtext = get_label(r)
    otext = get_label(o)
    s_article = get_article(s)
    if s_article is not None:
        sdoc = nlp(s_article['text'])
        for context in fuzzfind_plain(sdoc, stext, rtext, otext):
            yield context, s_article
    # todo: if object is a literal, then maybe we should search by (s, r) pair and literal's type
    # is_literal = (otext[0] == otext[-1] == '"')
    o_article = get_article(o)
    if o_article is not None:
        odoc = nlp(o_article['text'])
        for context in fuzzfind_plain(odoc, stext, rtext, otext):
            yield context, o_article


##### Test
jbtriples = list(gf.triples((dbr.JetBrains, dbo.product, None)))
def test(triples):
    for triple in jbtriples:
        ctxs = list(get_contexts(*triple))
        print(len(ctxs), triple, '\n')
        for i, (ctx, art) in enumerate(ctxs):
            print('_' * 40, i)
            print(ctx)

### Dataset management. key (input) points: output_path and valid_props.

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
def make_dataset(triples, output_path):
    log_total = 0
    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter, quotechar=quotechar, quoting=quoting)  # todo:adjust
        for i, triple in enumerate(triples):
            if validate(*triple):
                log.info('make_dataset: processing triple #{}: {}'.format(i, [str(t) for t in triple]))
                for j, (ctx, art) in enumerate(get_contexts(*triple)):
                    # write both the text and its' source
                    log_total += 1
                    log.info('make_dataset: contex #{} (total: {})'.format(j, log_total))
                    writer.writerow(list(triple) + [ctx.text.strip(), ctx.start, ctx.end] + [int(art['id'])])

def read_dataset(path, nlp):
    dataset = defaultdict(list)
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar, quoting=quoting)
        for s, r, o, ctext, cstart, cend, artid in reader:
            dataset[r].append(ctext)
            # yield (URIRef(s), URIRef(r), URIRef(o)), nlp(ctext)
    return dataset


# For every valid prop make a distinct file
for prop in valid_props:
    triples = gfall.triples((None, prop, None))
    make_dataset(triples, contexts_dir+'test2_{}.csv'.format(raw(prop)))

classes_file = props_dir + 'prop_classes.csv'
classes = {}
with open(classes_file, 'r', newline='') as f:
    reader = csv.reader(f, quotechar=csv.QUOTE_NONNUMERIC)
    for cls, rel, _ in reader:
        if int(cls) >= 0:
            classes[rel] = int(cls)


exit()

# Get classes for entities
classes = [dbo.Organisation, dbo.Software, dbo.ProgrammingLanguage, dbo.Person]
# oth_classes = [dbo.Company, dbo.VideoGame]

subjects = {}
for cls in classes:
    q = '''select distinct ?s from <http://dbpedia.org> from named <field> where { ?s rdf:type''' \
    + str(cls) + ''' . graph <field> {?s ?r ?o}}'''
    subjects[cls] = list(ds.query(q))


def query_raw(q):
    sparql = SPARQLWrapper(endpoint, update_endpoint)
    sparql.setHTTPAuth(DIGEST)
    sparql.setCredentials('dba', 'admin')
    sparql.setMethod(POST)
    sparql.setQuery(q)
    return sparql.query()
