import logging as log
import json
import os

from rdflib.graph import Dataset, Graph, ReadOnlyGraphAggregate
from rdflib.namespace import RDF, RDFS, OWL, FOAF, Namespace, URIRef
from rdflib.store import Store
from rdflib.plugins.stores.sparqlstore import SPARQLStore, SPARQLUpdateStore
from rdflib.plugins.stores.sparqlstore import SPARQLWrapper
from SPARQLWrapper import DIGEST, POST
from SPARQLWrapper.SPARQLExceptions import EndPointNotFound
from urllib.error import HTTPError

from experiments.utils import except_safe
from experiments.ontology.config import config


ont_config = config['ontology']
endpoint = update_endpoint = ont_config['endpoint']
store = SPARQLUpdateStore(endpoint, update_endpoint, autocommit=True)  # need to call store.commit explicitly todo: there's some trouble in sparqlstore's source with that autocommit logic
store.setHTTPAuth(DIGEST)
store.setCredentials(user=ont_config['endpoint_user'], passwd=ont_config['endpoint_passwd'])
ds = Dataset(store, default_union=False)

iri_dbo = 'http://dbpedia.org/ontology'
iri_dbpedia = 'http://dbpedia.org'
iri_labels = 'http://dbpedia.org/labels'
iri_redirects = 'http://dbpedia.org/redirects'
iri_disamb = 'http://dbpedia.org/disambiguations'
iri_field = 'field'
iri_more = 'field:more'
dbo = Namespace('http://dbpedia.org/ontology/')
dbr = Namespace('http://dbpedia.org/resource/')

# NB: for existing graphs use 'ds.get_context(iri)', for new graphs use 'ds.graph(iri)'
# errors or silent ignoring of actions (on non-existing graphs) otherwise should be expected...
# remove_graph also requires already existing graph
gdb = ds.get_context(iri_dbpedia)
gdbo = ds.get_context(iri_dbo)
glabels = ds.get_context(iri_labels)
gredirects = ds.get_context(iri_redirects)
gdisamb = ds.get_context(iri_disamb)
glo = ReadOnlyGraphAggregate([gdbo, glabels])
gall = ReadOnlyGraphAggregate([gdb, gdbo])
gf = ds.get_context(iri_field)
gmore = ds.get_context(iri_more)
gfall = ReadOnlyGraphAggregate([gf, gmore])


dir_articles = config['data']['articles_dir']


# It can happen that Virtuoso server is at the process of making a checkpoint, which will result in the following exceptions.
# Checkpoint takes few seconds, so, the easy way is just to wait few seconds and try again. Decorator does exactly that.
@except_safe(EndPointNotFound, HTTPError)
def get_article(subject):
    """Fail-safe when article is not present."""
    try:
        id_uri = next(gdb.objects(subject=subject, predicate=dbo.wikiPageID))
    except StopIteration:
        return None
    try:
        with open(os.path.join(dir_articles, id_uri)) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def raw(uri):
    return uri.rsplit('/', 1)[-1]


@except_safe(EndPointNotFound, HTTPError)
def get_label(uri: URIRef):
    t = list(glo.objects(uri, RDFS.label))
    t = raw(uri) if len(t) == 0 else str(t[0])
    return t.split('(')[0].strip(' _')  # remove disambiguations


@except_safe(EndPointNotFound, HTTPError)
def get_type(uri: URIRef):  # todo: add handling of literal types?
    t = list(gdb.objects(uri, RDF.type))
    return None if len(t) == 0 else t[0]


@except_safe(EndPointNotFound, HTTPError)
def get_fellow_redirects(uri: URIRef):
    # Uri can be the target page, to which others are redirected, or it can be itself redirected to something (or both)
    targets = list(gredirects.objects(subject=uri, predicate=dbo.wikiPageRedirects))
    if targets:  # if the uri is of the source
        uri = targets[0]  # then get the other sources (synonyms)
    sources = list(gredirects.subjects(object=uri, predicate=dbo.wikiPageRedirects))
    return sources


@except_safe(EndPointNotFound, HTTPError)
def get_fellow_disambiguations(uri: URIRef):
    # Uri is either the target page, to which others are disambiguated, or it is itself disambiguated to something
    sources = list(gdisamb.subjects(object=uri, predicate=dbo.wikiPageDisambiguates))
    if sources:  # if the uri is on of the disambiguations
        uri = sources[0]  # then get other uris disambiguated from the provided uri
    targets = list(gdisamb.objects(subject=uri, predicate=dbo.wikiPageDisambiguates))
    return targets


@except_safe(EndPointNotFound, HTTPError)
def get_superclass(uri):
    """
    :param uri: uri of the class to find superclass of
    :return: uri of superclass, @uri if class has no superclass, else None (e.g. @uri is not a uri of the class)
    """
    direct_scls = list(gdbo.objects(uri, RDFS.subClassOf))
    if len(direct_scls) > 0:
        c = direct_scls[0]
        return uri if c == OWL.Thing else c
    return None


class NERTypeResolver:
    from experiments.ontology.symbols import ENT_MAPPING
    # Map uris of final classes to its' names
    final_classes_names = {URIRef(dbo[s]): (ent_type if ent_type is not None else s) for s, ent_type in ENT_MAPPING.items()}

    def __init__(self, raw_classes=tuple()):
        assert(all(isinstance(x, URIRef) for x in raw_classes))
        self.superclasses_map = self.get_superclasses_map(raw_classes)

    def get_by_uri(self, uri, default_type=None):
        return self.get_by_type_uri(get_type(uri), default_type)

    def get_by_type_uri(self, type_uri, default_type=None):
        final_type_uri = self.superclasses_map.get(type_uri, self.get_final_class(type_uri))  # try to get the type from buffer
        if final_type_uri is not None:
            self.superclasses_map[type_uri] = final_type_uri  # add type to buffer (or do nothing useful if it is already there)
            return self.final_classes_names[final_type_uri]
        return default_type

    def get_final_class(self, cls):
        c = cls
        while not(c in self.final_classes_names or c is None):
            c = get_superclass(c)
            if c == cls:
                return None
            cls = c
        return c

    def get_superclasses_map(self, classes):
        superclasses = dict()
        for cls in classes:
            fc = self.get_final_class(cls)
            if fc is not None:
                superclasses[cls] = fc
        return superclasses


def query_raw(q):
    sparql = SPARQLWrapper(endpoint, update_endpoint)
    sparql.setHTTPAuth(DIGEST)
    sparql.setCredentials('dba', 'admin')
    sparql.setMethod(POST)
    sparql.setQuery(q)
    return sparql.query()


def test_redirects_disambigs():
    print('Disambiguates to targets')
    for uri in get_fellow_disambiguations(dbr.Alien):
        print(str(uri))
    print('Disambiguates from sources')
    for uri in get_fellow_disambiguations(dbr.Alien_Sun):
        print(str(uri))

    print('Redirects to targets')
    for uri in get_fellow_redirects(dbr.CDMA):
        print(str(uri))
    print('Redirects from sources')
    for uri in get_fellow_redirects(dbr.Code_division_multiple_access):
        print(str(uri))


def test_store_access():
    gtest = ds.get_context('gtest')
    gtest.update('INSERT DATA {<s1> <r1> <o1> }')
    # gtest.add((URIRef('s1'), URIRef('r1'), URIRef('o1')))  # analogous
    print(len(gtest.query('SELECT * WHERE {?s ?r ?o}')))
    gtest.update('DELETE DATA {<s1> <r1> <o1> }')
    print(len(gtest.query('SELECT * WHERE {?s ?r ?o}')))


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    test_redirects_disambigs()
    assert get_article(dbr.Microsoft) is not None
