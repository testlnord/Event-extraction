import spacy

from experiments.ontology.symbols import POS_TAGS, DEP_TAGS
from experiments.tags import CategoricalTags

# mappers from tags to classes
#   num of classes: POS, dep,

def shortest_dep_path(span1, span2):
    """Find shortest path that connects two subtrees (span1 and span2) in the dependency tree"""
    path = []
    ancestors1 = list(span1.root.ancestors)
    for anc in span2.root.ancestors:
        path.append(anc)
        # If we find the nearest common ancestor
        if anc in ancestors1:
            # Add to common path subpath from span1 to common ancestor
            edge = ancestors1.index(anc)
            path.extend(ancestors1[:edge])
            break
    # Sort to match sentence order
    path = list(sorted(path, key=lambda token: token.i))
    return path

buf = {}
# todo: make some buffer for charmap
def chars2spans(doc, *char_offsets_pairs):
    charmap = {t.idx: t.i for t in doc}
    charmap.update({t.idx+len(t): t.i+1 for t in doc})
    return [doc[charmap[a]:charmap[b]] for a, b in char_offsets_pairs]

nlp = spacy.load('en')
pos_tags = CategoricalTags(POS_TAGS)
dep_tags = CategoricalTags(DEP_TAGS)
# todo: do something with the interface?
def encode(s, r, o, s0, s1, o0, o1, ctext, cstart, cend, artid):
    doc = nlp(ctext)
    sent, s_span, o_span = chars2spans(doc, [(cstart, cend), (s0, s1), (o0, o1)])
    sdp = shortest_dep_path(s_span, o_span)
    _pos_tags = [pos_tags.encode(t.pos_) for t in sdp]
    _dep_tags = [dep_tags.encode(t.dep_) for t in sdp]
    vectors = [t.vector for t in sdp]
    return _pos_tags, _dep_tags, vectors

