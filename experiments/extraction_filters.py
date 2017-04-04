from experiments.extraction import Extraction2


#### Purpose of all of it is Named Entities Candidates search

class NamedEntitiesFilter:
    """
    In general, enhance precision of extracted named entities.
    (e.g. add version numbers to software mentions)
    """
    # todo: use IOB ner tags from spacy
    # todo: heuristic for like "version xx OF yy" -- add yy to relation
    # or maybe "xx OF yy"
    # todo: process conjugated named entities
    pass

class CoherenceFilter:
    """
    Remove incoherent extractions
    """
    # todo: use language model?
    # todo: use confidence from generic KB?
    pass

class DomainFilter:
    # todo: filter out extractions without named entities
    # todo: filter out e-s without named enitites from our domain based on existing KB (taxonomy, etc.), using hand-crafted rules, I guess.
    pass

class DuplicatesFilter:
    """
    Choose the best distinct extractions from all the provided extractions.
    """
    pass
