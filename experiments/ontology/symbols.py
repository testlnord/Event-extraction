

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
    "Work": "PRODUCT",
    "ProgrammingLanguage": None,
    # "ProgrammingLanguage": "LANGUAGE",
    "Software": None,
    # "Software": "PRODUCT",
    "VideoGame": None,
    # "VideoGame": "WORK_OF_ART",
}


NEW_ENT_CLASSES = {db_ent for db_ent, spacy_ent in ENT_MAPPING.items() if spacy_ent is None}
ENT_CLASSES = NEW_ENT_CLASSES.union(set(filter(None, ENT_MAPPING.values())))  # classes we interested in

LESS_ENT_CLASSES = NER_TAGS_LESS.union(NEW_ENT_CLASSES)
ALL_ENT_CLASSES = NER_TAGS.union(NEW_ENT_CLASSES)


RC_CLASSES_MAP = {
    'http://dbpedia.org/ontology/developer': 'author',
    'http://dbpedia.org/ontology/designer': 'author',
    'http://dbpedia.org/ontology/author': 'author',
    'http://dbpedia.org/ontology/publisher': 'author',
    'http://dbpedia.org/ontology/distributor': 'author',
    'http://dbpedia.org/ontology/knownFor': 'knownFor',  # bad class
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


# Different class schema
RC_CLASSES_MAP_ALL2 = {
    'http://dbpedia.org/ontology/developer': 'author',
    'http://dbpedia.org/ontology/designer': 'author',
    'http://dbpedia.org/ontology/author': 'author',
    'http://dbpedia.org/ontology/creator': 'author',
    'http://dbpedia.org/ontology/publisher': 'author',
    'http://dbpedia.org/ontology/distributor': 'author',
    # 'http://dbpedia.org/ontology/knownFor': 'knownFor',
    'http://dbpedia.org/ontology/product': 'product',
    'http://dbpedia.org/ontology/computingPlatform': 'computingPlatform',
    'http://dbpedia.org/ontology/location': 'location',
    'http://dbpedia.org/ontology/locationCity': 'location',
    'http://dbpedia.org/ontology/locationCountry': 'location',
    'http://dbpedia.org/ontology/foundationPlace': 'location',

    # 'http://dbpedia.org/ontology/keyPerson': 'keyPerson',
    'http://dbpedia.org/ontology/foundedBy': 'founder',
    'http://dbpedia.org/ontology/founder': 'founder',
    'http://dbpedia.org/ontology/owner': 'owner',
    'http://dbpedia.org/ontology/parentCompany': 'owner',
    'http://dbpedia.org/ontology/owningCompany': 'owner',
    'http://dbpedia.org/ontology/subsidiary': 'childEntity',
    'http://dbpedia.org/ontology/division': 'childEntity',
    'http://dbpedia.org/ontology/predecessor': 'predecessor',
    'http://dbpedia.org/ontology/successor': 'successor',
}


# Different class schema
RC_CLASSES_MAP_ALL3 = {
    'http://dbpedia.org/ontology/developer': 'author',
    'http://dbpedia.org/ontology/designer': 'author',
    'http://dbpedia.org/ontology/author': 'author',
    'http://dbpedia.org/ontology/creator': 'author',
    'http://dbpedia.org/ontology/publisher': 'author',
    'http://dbpedia.org/ontology/distributor': 'author',
    # 'http://dbpedia.org/ontology/knownFor': 'knownFor',
    'http://dbpedia.org/ontology/product': 'product',
    'http://dbpedia.org/ontology/computingPlatform': 'computingPlatform',
    'http://dbpedia.org/ontology/location': 'location',
    'http://dbpedia.org/ontology/locationCity': 'location',
    'http://dbpedia.org/ontology/locationCountry': 'location',
    'http://dbpedia.org/ontology/foundationPlace': 'location',

    # 'http://dbpedia.org/ontology/keyPerson': 'keyPerson',
    # 'http://dbpedia.org/ontology/foundedBy': 'founder',
    # 'http://dbpedia.org/ontology/founder': 'founder',
    # 'http://dbpedia.org/ontology/owner': 'owner',
    # 'http://dbpedia.org/ontology/parentCompany': 'owner',
    # 'http://dbpedia.org/ontology/owningCompany': 'owner',
    # 'http://dbpedia.org/ontology/subsidiary': 'childEntity',
    # 'http://dbpedia.org/ontology/division': 'childEntity',
    # 'http://dbpedia.org/ontology/predecessor': 'predecessor',
    # 'http://dbpedia.org/ontology/successor': 'successor',
}


# keyActor and parentEntity are similar sometimes
RC_INVERSE_MAP = {
    'successor': 'predecessor',
    'predecessor': 'successor',
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
    # "Other": "No relation",
}


SEMEVAL_CLASSES_MAP = {c: c for c in SEMEVAL_CLASSES}


KBP37_CLASSES = {
    "per:alternate_names",
    "per:origin",
    "per:spouse",
    "per:title",
    "per:employee_of",
    "per:countries_of_residence",
    "per:stateorprovinces_of_residence",
    "per:cities_of_residence",
    "per:country_of_birth",
    "org:alternate_names",
    "org:subsidiaries",
    "org:top_members/employees",
    "org:founded",
    "org:founded_by",
    "org:country_of_headquarters",
    "org:stateorprovince_of_headquarters",
    "org:city_of_headquarters",
    "org:members",
    # "no_relation",  # anyway will be mapped to default
}


KBP37_CLASSES_MAP = {c: c for c in KBP37_CLASSES}


WORDNET_HYPERNYM_CLASSES = {
    'adj.all',
    'adj.pert',
    'adj.ppl',
    'adv.all',
    'noun.Tops',
    'noun.act',
    'noun.animal',
    'noun.artifact',
    'noun.attribute',
    'noun.body',
    'noun.cognition',
    'noun.communication',
    'noun.event',
    'noun.feeling',
    'noun.food',
    'noun.group',
    'noun.location',
    'noun.motive',
    'noun.object',
    'noun.person',
    'noun.phenomenon',
    'noun.plant',
    'noun.possession',
    'noun.process',
    'noun.quantity',
    'noun.relation',
    'noun.shape',
    'noun.state',
    'noun.substance',
    'noun.time',
    'verb.body',
    'verb.change',
    'verb.cognition',
    'verb.communication',
    'verb.competition',
    'verb.consumption',
    'verb.contact',
    'verb.creation',
    'verb.emotion',
    'verb.motion',
    'verb.perception',
    'verb.possession',
    'verb.social',
    'verb.stative',
    'verb.weather'
}
