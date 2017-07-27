import logging as log
from itertools import chain

from intervaltree import IntervalTree, Interval
import spacy
from nltk.corpus import wordnet as wn


def is_connected(span):
    """Check if the dependency tree of a span is contiguous (i.e. it is a tree, not a forest)"""
    r = span.root
    return (r.left_edge.i == 0) and (r.right_edge.i == len(span)-1)
    # return len(r.subtree) == len(span.subtree)  # another way


def expand_ents(token_seq, expand_noun_chunks=False):
    """
    Expand noun chunks and ents (add them with tokens to the new list) from underlying doc. Assuming all tokens are from the same doc
    :param token_seq: sequence of tokens (i.e. list)
    :return: list of tokens, including tokens from @token_seq
    """
    doc = token_seq[0].doc
    path = list(token_seq)
    inds = set(token.i for token in path)
    for span in (chain(doc.ents, doc.noun_chunks) if expand_noun_chunks else doc.ents):
        s_inds = set(t.i for t in span)
        if not inds.isdisjoint(s_inds):
            inds.update(s_inds)
    tokens = [doc[i] for i in sorted(inds)]
    return tokens


# todo: context subtrees around key spans?
def shortest_dep_path(span1, span2, include_spans=True, nb_context_tokens=0):
    """
    Find shortest path that connects two subtrees (span1 and span2) in the dependency tree.
    :param span1: one of the spans
    :param span2: one of the spans
    :param include_spans: whether include source spans in the SDP or not
    :param nb_context_tokens: number of tokens for expanding SDP to the right and to the left (default 0, i.e. plain SDP)
    :return: list of tokens
    """
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

    if include_spans:
        path = list(span1) + path + list(span2)

    # Sort to match the sentence order
    path = list(sorted(path, key=lambda token: token.i))

    # Extract context in the dep tree for border tokens in path
    left = path[0]
    lefts = list(left.lefts)
    lefts = lefts[max(0, len(lefts) - nb_context_tokens):]  # todo: duplicates on the left sometimes appear
    right = path[-1]
    rights = list(right.rights)
    lr = len(rights)
    rights = rights[:min(lr, nb_context_tokens)]
    return list(lefts) + path + list(rights)


def chars2spans(doc, *char_offsets_pairs):
    l = len(doc)-1
    lt = len(doc.text)+1
    charmap = IntervalTree(Interval(doc[i].idx, doc[i+1].idx, i) for i in range(l))
    charmap.addi(doc[-1].idx, lt, l)
    charmap.addi(lt, lt+1, l+1)  # as offsets_pairs are not-closed intervals, we need it for tokens at the end
    # tmp = [(t.idx, t.i) for t in doc]
    # charmap = dict([(i, ti) for k, (idx, ti) in enumerate(tmp[:-1]) for i in range(idx, tmp[k+1][0])])

    # def ii(p):
    #     i = charmap[p].pop()
    #     return i.data + int(p != i.begin)  # handle edge case when point falls into the previous interval
    # return [doc[ii(a):ii(b)] for a, b in char_offsets_pairs]
    def ii(p): return charmap[p].pop().data  # help function for convenience
    slices = []
    for a, b in char_offsets_pairs:
        ia = ii(a)
        ib = ii(b)
        ib += int(ib == ia)  # handle edge case when point falls into the previous interval which results in empty span
        slices.append(doc[ia:ib])
    return slices


wordnet_pos_allowed = {
    "NOUN": wn.NOUN,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
    "VERB": wn.VERB,
}


def get_hypernym(nlp, token):
    pos = wordnet_pos_allowed.get(token.pos_)
    if pos is not None:
        synsets = wn.synsets(token.text, pos)
        if len(synsets) > 0:
            hs = synsets[0].hypernyms()
            h = hs[0] if len(hs) > 0 else synsets[0]
            raw = h.lemma_names()[0]
            return nlp.vocab[raw]


