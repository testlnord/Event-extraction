from __future__ import unicode_literals

DEP_TAGS = {
    "acomp",
    "advcl",
    "advmod",
    "agent",
    "amod",
    "appos",
    "attr",
    "aux",
    "auxpass",
    "cc",
    "ccomp",
    "complm",
    "conj",
    "cop", # U20
    "csubj",
    "csubjpass",
    "dep",
    "det",
    "dobj",
    "expl",
    "hmod",
    "hyph",
    "infmod",
    "intj",
    "iobj",
    "mark",
    "meta",
    "neg",
    "nmod",
    "nn",
    "npadvmod",
    "nsubj",
    "nsubjpass",
    "num",
    "number",
    "oprd",
    "obj", # U20
    "obl", # U20
    "parataxis",
    "partmod",
    "pcomp",
    "pobj",
    "poss",
    "possessive",
    "preconj",
    "prep",
    "prt",
    "punct",
    "quantmod",
    "rcmod",
    "root",
    "xcomp",
}


DEP_TAGS.update([  # not documented tags
    'acl',
    'case',
    'compound',
    'dative',
    'nummod',
    'relcl',
    'ROOT',
    'predet'
])


POS_TAGS = {
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CONJ",
    "CCONJ", # U20
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
    "EOL",
    "SPACE",
}


IOB_TAGS = {'I', 'O', 'B'}


NER_TAGS = {
    "PERSON",
    "NORP",
    "FACILITY",  # possibly it is not present; 'FAC' used instead
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LANGUAGE",

    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
}

# Undocumented, but present in EntityRecogniser model
NER_TAGS.update([
    'FAC',
    'LAW',
])


NER_TAGS_LESS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "LANGUAGE",
    "EVENT",

    "DATE",
    "TIME",
    "MONEY",
    "PERCENT",
    "ORDINAL",
    "CARDINAL",
}


ENT_MAPPING = {
    "Person": "PERSON",
    "Organisation": "ORG",
    "Place": "LOC",
    "Settlement": "GPE",
    "Country": "GPE",
    "Language": "LANGUAGE",
    "ProgrammingLanguage": None,
    "Work": "PRODUCT",
    "Software": None,
    "VideoGame": None,
}


NEW_ENT_CLASSES = {db_ent for db_ent, spacy_ent in ENT_MAPPING.items() if spacy_ent is None}
ENT_CLASSES = NEW_ENT_CLASSES.union(set(filter(None, ENT_MAPPING.values())))

LESS_ENT_CLASSES = NER_TAGS_LESS.union(NEW_ENT_CLASSES)
ALL_ENT_CLASSES = NER_TAGS.union(NEW_ENT_CLASSES)


RC_CLASSES_MAP = {
    'http://dbpedia.org/ontology/developer':'author',
    'http://dbpedia.org/ontology/designer':'author',
    'http://dbpedia.org/ontology/author':'author',
    'http://dbpedia.org/ontology/publisher':'author',
    'http://dbpedia.org/ontology/distributor':'author',
    'http://dbpedia.org/ontology/knownFor':'knownFor',
    'http://dbpedia.org/ontology/product':'product',
    'http://dbpedia.org/ontology/computingPlatform':'computingPlatform',
    'http://dbpedia.org/ontology/keyPerson':'keyActor',
    'http://dbpedia.org/ontology/foundedBy':'keyActor',
    'http://dbpedia.org/ontology/founder':'keyActor',
    'http://dbpedia.org/ontology/owner':'keyActor',
    'http://dbpedia.org/ontology/location':'location',
    'http://dbpedia.org/ontology/locationCity':'location',
    'http://dbpedia.org/ontology/locationCountry':'location',
    'http://dbpedia.org/ontology/foundationPlace':'location',
}


RC_CLASSES_MAP_MORE = {
    'http://dbpedia.org/ontology/parentCompany':'parentEntity',
    'http://dbpedia.org/ontology/owningCompany':'parentEntity',
    'http://dbpedia.org/ontology/predecessor':'parentEntity',
    'http://dbpedia.org/ontology/subsidiary':'childEntity',
    'http://dbpedia.org/ontology/division':'childEntity',
    'http://dbpedia.org/ontology/successor':'childEntity',
}


RC_CLASSES_MAP_ALL = RC_CLASSES_MAP.copy()
RC_CLASSES_MAP_ALL.update(RC_CLASSES_MAP_MORE)

RC_CLASSES = set(RC_CLASSES_MAP.values())
RC_CLASSES_ALL = set(RC_CLASSES_MAP_ALL.values())
