import urllib.request
import time
from bs4 import BeautifulSoup
from rdflib import Dataset, Graph
from rdflib.plugins.stores.sparqlstore import SPARQLStore, SPARQLUpdateStore
from SPARQLWrapper import DIGEST

endpoints = ['http://dbpedia.org/sparql', 'http://oat.openlinksw.com/sparql']
rs_table_page = '?help=rdfinf'
other_rs = []
rule_sets = {}
rs_names = {}


q = '''construct where {?r ?s ?o}'''
def getq(rs_uri):
    return '''construct where { graph <''' + rs_uri + '''> {?r ?s ?o}}'''

def get_rule_sets(url):
    bpage = urllib.request.urlopen(url).read().decode('utf8')
    soup = BeautifulSoup(bpage, 'html.parser')
    table = soup.find('table')
    rows = table.findAll('tr')
    for row in rows:
        ss = [s for col in row.findAll('td') for s in col.findAll(text=True)]
        if len(ss) == 2:
            rs_name, rs_uri = ss
            yield rs_name, rs_uri


def fetch(endpoint, timeout=0):
    store = SPARQLStore(endpoint)
    ds = Dataset(store)
    for rs_name, rs_uri in get_rule_sets(endpoint + rs_table_page):
        # TODO: maybe do not discrad but try to merge? no.
        if rs_uri not in rule_sets:
            # TODO: handle possible query error?
            gr = ds.get_context(rs_uri)
            try:
                rs_triples = gr.query(q)
                yield rs_name, rs_uri, rs_triples
                time.sleep(timeout)
            except:
                print('error with', rs_uri)
                other_rs.append(rs_uri)

def get_ds0():
    update_endpoint = 'http://localhost:8890/sparql-auth'
    # query_endpoint = 'http://localhost:8890/sparql'
    store = SPARQLUpdateStore(update_endpoint, update_endpoint, autocommit=True)
    store.setHTTPAuth(DIGEST)
    store.setCredentials(user='dba', passwd='admin')
    return Dataset(store)


command_tmpl = 'rdfs_rule_set({}, {});'
commands = []
outdir = '/home/user/datasets/rule_sets/'
for endpoint in endpoints:
    print(endpoint)
    for rs_name, rs_uri, qres in fetch(endpoint, 0):
        length = len(qres)
        print(rs_name, rs_uri, length)
        if length < 10000 and rs_uri not in rule_sets:
            rule_sets[rs_uri] = qres
            rs_names[rs_uri] = rs_name
            commands.append(command_tmpl.format(rs_name, rs_uri))
        else:
            other_rs.append(rs_uri)

with open(outdir + 'commands', 'w') as f:
    f.writelines(commands)

ds0 = get_ds0()
for rs_uri, qres in rule_sets.items():
    ng = ds0.get_context(rs_uri)
    qlen = len(qres)
    glen = len(ng)
    print('qlen:', qlen, 'glen', glen, '|', rs_uri)  # for user: to see unempty graphs

    filename = outdir + rs_names[rs_uri].split('/')[-1].strip('#') + '.ttl'  # possible (but unlikely) collisions of filenames
    print(filename)
    gtmp = Graph()
    for t in qres:
        gtmp.add(t)
    gtmp.serialize(filename, format='turtle')
    with open(filename + '.graph', 'w') as f:
        f.write(rs_uri)



