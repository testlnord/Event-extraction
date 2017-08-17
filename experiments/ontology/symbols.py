

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


# On the right should be only known to Spacy entity types (e.g. see NER_TAGS above)
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
    'http://dbpedia.org/ontology/developer': 'author',
    'http://dbpedia.org/ontology/designer': 'author',
    'http://dbpedia.org/ontology/author': 'author',
    'http://dbpedia.org/ontology/publisher': 'author',
    'http://dbpedia.org/ontology/distributor': 'author',
    'http://dbpedia.org/ontology/knownFor': 'knownFor',
    'http://dbpedia.org/ontology/product': 'product',
    'http://dbpedia.org/ontology/computingPlatform': 'computingPlatform',
    'http://dbpedia.org/ontology/keyPerson': 'keyActor',
    'http://dbpedia.org/ontology/foundedBy': 'keyActor',
    'http://dbpedia.org/ontology/founder': 'keyActor',
    'http://dbpedia.org/ontology/owner': 'keyActor',
    'http://dbpedia.org/ontology/location': 'location',
    'http://dbpedia.org/ontology/locationCity': 'location',
    'http://dbpedia.org/ontology/locationCountry': 'location',
    'http://dbpedia.org/ontology/foundationPlace': 'location',
}


RC_CLASSES_MAP_MORE = {
    'http://dbpedia.org/ontology/parentCompany': 'parentEntity',
    'http://dbpedia.org/ontology/owningCompany': 'parentEntity',
    'http://dbpedia.org/ontology/predecessor': 'parentEntity',
    'http://dbpedia.org/ontology/subsidiary': 'childEntity',
    'http://dbpedia.org/ontology/division': 'childEntity',
    'http://dbpedia.org/ontology/successor': 'childEntity',
}


RC_CLASSES_MAP_ALL = RC_CLASSES_MAP.copy()
RC_CLASSES_MAP_ALL.update(RC_CLASSES_MAP_MORE)


RC_CLASSES = set(RC_CLASSES_MAP.values())
RC_CLASSES_ALL = set(RC_CLASSES_MAP_ALL.values())

# keyActor and parentEntity are similar sometimes
RC_INVERSE_MAP = {
    'childEntity': 'parentEntity',
    'parentEntity': 'childEntity',
    'author': 'product',
    'product': 'author',
}


# With descriptions
SEMEVAL_CLASSES = {
    "Cause-Effect": "X is the cause of Y",
    "Entity-Origin": "Y is the origin of an entity X, and X is coming or derived from that origin.",
    "Message-Topic": "X is a communicative message containing information about Y",
    "Product-Producer": "X is a product of Y",
    "Entity-Destination": "Y is the destination of X in the sense of X moving toward Y",
    "Member-Collection": "X is a member of Y",
    "Instrument-Agency": "X is the instrument (tool) of Y or Y uses X",
    "Component-Whole": "X has an operating or usable purpose within Y",
    "Content-Container": "X is or was stored or carried inside Y",
}


SEMEVAL_CLASSES_MAPPED = {
    # "Cause-Effect": "",
    "Entity-Origin": "location",
    # "Message-Topic": "",
    "Product-Producer": "author",
    # "Entity-Destination": "",
    # "Member-Collection": "",
    # "Instrument-Agency": "",
    "Component-Whole": "parentEntity",
    # "Content-Container": "",
}


KBP37_CLASSES = {
    "per:alternate names",
    "per:origin",
    "per:spouse",
    "per:title",
    "per:employee of",
    "per:countries of residence",
    "per:stateorprovinces of residence",
    "per:cities of residence",
    "per:country of birth",
    "org:alternate names",
    "org:subsidiaries",
    "org:top members/employees",
    "org:founded",
    "org:founded by",
    "org:country of headquarters",
    "org:stateorprovince of headquarters",
    "org:city of headquarters",
    "org:members",
    "no relation",
}


KBP37_CLASSES_MAPPED = {
    # "per:alternate names",
    # "per:origin",
    # "per:spouse",
    # "per:title",
    "per:employee of": "",  # todo: inverse to keyActor
    # "per:countries of residence",
    # "per:stateorprovinces of residence",
    # "per:cities of residence",
    # "per:country of birth",
    # "org:alternate names",
    "org:subsidiaries": "childEntity",
    "org:top members/employees": "keyActor",
    "org:founded": "keyActor",
    "org:founded by": "keyActor",
    "org:country of headquarters": "location",
    "org:stateorprovince of headquarters": "location",
    "org:city of headquarters": "location",
    "org:members": "",  # todo: understand
    # "no relation": None,
}
