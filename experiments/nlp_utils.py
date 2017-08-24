import logging as log
from itertools import chain

from cytoolz.itertoolz import groupby
from intervaltree import IntervalTree, Interval
from nltk.corpus import wordnet as wn


def merge_ents_offsets(primal_ents, other_ents):
    """
    Merge ent lists with non-overlapping entries, giving precedence to primal_ents.
    NB: all ents must have attributes 'start_char' and 'end_char'
    :param primal_ents: iterable of tuples of form (begin_offset, end_offset, data)
    :param other_ents: iterable of tuples of form (begin_offset, end_offset, data)
    :return: merged list of ents
    """
    offsets = [(e.start_char, e.end_char) for e in primal_ents if e.start_char < e.end_char]
    ents_tree = IntervalTree.from_tuples(offsets)
    ents_filtered = [ent for ent in other_ents if not ents_tree.overlaps(ent.start_char, ent.end_char)]
    ents_filtered.extend(primal_ents)
    return ents_filtered


def sentences_ents(doc, ents=None):
    """
    Group doc.ents by sentences.
    :param doc: spacy.Doc
    :param ents: iterable of objects with attributes 'start_char' and 'end_char' (if None (default), use doc.ents)
    :yield: Tuple[spacy.token.Span, List[spacy.token.Span]]
    """
    # Group entities by sentences
    sents_bound_tree = IntervalTree.from_tuples([(s.start_char, s.end_char, i) for i, s in enumerate(doc.sents)])
    # Help function for convenience
    def index(ent):
        # print(list(sorted([tuple(i) for i in sents_bound_tree.all_intervals], key=lambda t: t[0])))
        # print(ent)
        sents = sents_bound_tree[ent.start_char: ent.end_char]
        if sents: return sents.pop().data

    if ents is None: ents = doc.ents
    ents_in_sents = groupby(index, ents)
    for i, sent in enumerate(doc.sents):
        sent_ents = ents_in_sents.get(i, list())
        yield sent, sent_ents


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
    :return: (list of tokens, index of common ancestor in that list OR -1 if it wasn't found, ispan1, ispan2)
    """
    path = []
    ancestors1 = list(span1.root.ancestors)
    common_anc = None
    for anc in span2.root.ancestors:
        path.append(anc)
        # If we find the nearest common ancestor
        if anc in ancestors1:
            common_anc = anc
            # Add to common path subpath from span1 to common ancestor
            edge = ancestors1.index(anc)
            path.extend(ancestors1[:edge])
            break

    if include_spans:
        path = list(span1) + path + list(span2)

    # Remove possible duplicates and sort to match the sentence order
    uniques = {token.i: token for token in path}
    path = [uniques[i] for i in sorted(uniques)]

    # Extract context in the dep tree for border tokens in path
    left = path[0]
    lefts = list(left.lefts)
    lefts = lefts[max(0, len(lefts) - nb_context_tokens):]
    right = path[-1]
    rights = list(right.rights)
    lr = len(rights)
    rights = rights[:min(lr, nb_context_tokens)]
    path = list(lefts) + path + list(rights)

    iroot = path.index(common_anc) if common_anc in path else -1
    return path, iroot


def chars2spans(doc, *char_offsets_pairs):
    """
    Transform char offsets pairs to spans in doc. Return None if
    :param doc:
    :param char_offsets_pairs:
    :return: List[spacy.token.Span] or None
    """
    l = len(doc)-1
    lt = len(doc.text)+1
    charmap = IntervalTree(Interval(doc[i].idx, doc[i+1].idx, i) for i in range(l))
    charmap.addi(doc[-1].idx, lt, l)
    charmap.addi(lt, lt+1, l+1)  # as offsets_pairs are not-closed intervals, we need it for tokens at the end
    # tmp = [(t.idx, t.i) for t in doc]
    # charmap = dict([(i, ti) for k, (idx, ti) in enumerate(tmp[:-1]) for i in range(idx, tmp[k+1][0])])

    def ii(p):
        _res = charmap[p]  # help function for convenience
        if _res:
            return _res.pop().data

    slices = []
    for a, b in char_offsets_pairs:
        ia = ii(a)
        ib = ii(b)
        if ia is not None and ib is not None:
            ib += int(ib == ia)  # handle edge case when point falls into the previous interval which results in empty span
            slices.append(doc[ia:ib])
        else:
            log.warning('chars2spans: span ({}, {}) is not in the IntervalTree: {}'.format(a, b, charmap.all_intervals))
            return None  # fail if any of the spans fails
    return slices


wordnet_pos_allowed = {
    "NOUN": wn.NOUN,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
    "VERB": wn.VERB,
}


def get_hypernym_raw(token):
    pos = wordnet_pos_allowed.get(token.pos_)
    if pos is not None:
        synsets = wn.synsets(token.text, pos)
        if synsets:
            hs = synsets[0].hypernyms()
            return hs[0] if hs else synsets[0]


def get_hypernym(nlp, token):
    h = get_hypernym_raw(token)
    if h:
        raw = h.lemma_names()[0].replace('_', ' ')  # wordnet representation uses '_' instead of space
        return nlp(raw)  # wordnet lemmas can consist out of several words, so, nlp.vocab[raw] is not suitable


def get_hypernym_cls(token):
    h = get_hypernym_raw(token)
    if h: return h.lexname()


def get_hypernym_classes(nlp_vocab):
    return {synset.lexname() for lexem in nlp_vocab for synset in wn.synsets(lexem.text)}
