from __future__ import unicode_literals

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

NER_TAGS = [
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
]

# Undocumented, but present in EntityRecogniser model
NER_TAGS += [
    'FAC',
    'LAW',
]

IOB_TAGS = ['I', 'O', 'B']

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

NEW_ENT_CLASSES = [db_ent for db_ent, spacy_ent in ENT_MAPPING.items() if spacy_ent is None]
ENT_CLASSES = list(filter(None, set(ENT_MAPPING.values()))) + NEW_ENT_CLASSES
ALL_ENT_CLASSES = NER_TAGS + NEW_ENT_CLASSES

DEP_TAGS = [
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
]

DEP_TAGS += [  # not documented tags
    'acl',
    'case',
    'compound',
    'dative',
    'nummod',
    'relcl',
    'ROOT',
    'predet'
]
