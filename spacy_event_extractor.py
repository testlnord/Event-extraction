import re
import types
from itertools import *

import spacy.tokens
import sys
from spacy.en import English

from data_mining.downloaders import article_downloaders
from db.db_handler import DatabaseHandler
from event import Event


class SpacyEventExtractor:
    _nlp = English()
    _keywords = list(map(lambda s: s.strip().lower(), open('keywords.txt', 'r').readlines()))
    _known_phrases = [('is out', 'released'), ('is here', 'released'), ('is there', 'released'),
                      ('is out', 'released'), ('is open', 'started'), ('is available', 'released'),
                      ('please welcome', 'we released')
                     ]
    _important_actions = ['release', 'start', 'publish', 'announce', 'update']

    def __init__(self):
        pass

    @staticmethod
    def _have_pronouns(text: str) -> bool:
        pronouns = ['i', 'you', 'he', 'she', 'they', 'be', 'him', 'her', 'it']
        # 'we' is a good pronoun as it refers to a company
        return list(filter(lambda s: s.lower() in pronouns, text.split())) != []

    @staticmethod
    def _is_present_simple(verb: spacy.tokens.Token) -> bool:
        for child in verb.children:
            if child.orth_ == 'will':
                return False  # will have etc
        lemma = verb.lemma_.lower()
        if verb.orth_.lower() in [lemma, lemma + 's', lemma + 'es', 'have', 'has', 'do', 'is', 'are']:
            return True
        return False

    @staticmethod
    def _is_present_continuous(verb: spacy.tokens.Token) -> bool:
        for child in verb.children:
            if child.dep_ == 'aux' and child.lemma_ not in ['be', 'is', 'are', 'am']:
                return False  # will have etc
        return verb.orth_.endswith('ing')

    @staticmethod
    def _get_tree(root: spacy.tokens.Token, depth: int, token_filter: types.FunctionType) -> [spacy.tokens.Token]:
        """Get list of tokens dependent on given root and satisfying given token_filter"""
        if depth == 0:
            return [root] if token_filter(root) else []

        result = []
        # for tokens on the left of the root, whose head is root
        for child in filter(token_filter, root.lefts):
            result += SpacyEventExtractor._get_tree(child, depth - 1, token_filter)
        result.append(root)
        # for tokens on the right of the root, whose head is root
        for child in filter(token_filter, root.rights):
            result += SpacyEventExtractor._get_tree(child, depth - 1, token_filter)
        return result

    @staticmethod
    def _get_chunk(token: spacy.tokens.Token) -> str:
        """Get string representation of a chunk.
        Chunk is one or more tokens that forms semantic unit.
        For example, compound tokens or tokens with dependent tokens."""

        if token is None:
            return ""

        def token_filter(tok):
            """True for various modifiers of tok and compound tokens, which include tok"""
            return tok is token or \
                   tok.dep_.endswith("mod") or \
                   tok.dep_ == "compound"

        tree = SpacyEventExtractor._get_tree(root=token, depth=2, token_filter=token_filter)
        return " ".join(map(str, tree))

    @staticmethod
    def _get_prep_with_word(token: spacy.tokens.Token) -> (str, spacy.tokens.Token):
        """Get prepositional modifiers of the token and important perposition's child"""
        if token is None:
            return "", None

        prep = None
        # search of prepositions
        for child in token.rights:
            if child.dep_ == "prep":
                prep = child
                break
        if prep is None:
            return "", None

        for word in prep.children:
            # if preposition has child of type 'object of preposition' or 'complement of a preposition'
            # then add it to the result
            if word.dep_ in ["pobj", "pcomp"]:
                chunk_str = SpacyEventExtractor._get_chunk(word)
                return str(prep) + " " + chunk_str, word

        return "", None

    @staticmethod
    def _get_full_entity(entity: spacy.tokens.Token) -> str:
        """Get entity token with all related tokens (i.e. prepositional modifiers)
        so, we are extracting such token tree with entity
        entity
            mod & compound
                mod & compound
            prep
                pobj | pcomp
                    mod & compound
                        mod & compound
                    (repeat)
                    prep
                        pobj | pcomp
                            mod & compound
                                mod & compound
                            (repeat)
                            ...
        """
        entity_string = SpacyEventExtractor._get_chunk(entity)

        word = entity
        while True:
            prep, word = SpacyEventExtractor._get_prep_with_word(word)
            if word is None:
                break
            entity_string += " " + prep
        return entity_string

    @staticmethod
    def _replace_we(replace_we, string):
        """Replace pronoun 'we' in string with string 'replace_we'"""
        new_string = ""
        for word in string.split():
            if word == "we" and replace_we is not None:
                new_string += replace_we + " "
            elif word == "We" and replace_we is not None:
                new_string += replace_we.capitalize() + " "
            else:
                new_string += str(word) + " "
        return new_string

    @staticmethod
    def _remove_extra_whitespaces(text):
        return ' '.join(text.strip().split())

    @staticmethod
    def _get_entity1(span):
        """Get nominal subject of the span's root, if there is one"""
        for word in span:
            if word.head is word: # main verb
                for child in word.children:
                    if child.dep_.endswith("nsubj"):
                        return child
                break
        return None

    @staticmethod
    def _get_action(verb):
        """Get auxiliary verbs of the given verb and the verb itself"""
        aux_verbs = ""
        for child in verb.children:
            if child.dep_ == "aux" or child.dep_ == "neg":
                aux_verbs += str(child)
        return SpacyEventExtractor._remove_extra_whitespaces(str(aux_verbs) + ' ' + str(verb))

    @staticmethod
    def _get_entity2(verb):
        """Get direct object of the given verb, if there is one"""
        for child in verb.children:
            if child.dep_ == "dobj":
                return child
        return None

    @staticmethod
    def extract(text: str, replace_we: str = None) -> [Event]:

        # just because sometimes spaCy fails on sth like we've
        for aux, replace_with in [('ve', 'have'), ('re', 'are')]:
            text = text.replace("'" + aux, " " + replace_with).replace("’" + aux, " " + replace_with)

        # replacing known_phrases
        for abbr, full in SpacyEventExtractor._known_phrases:
            reg = re.compile(abbr, re.IGNORECASE)
            text = reg.sub(full, text)

        if len(text) == 0:
            return []

        text_doc = SpacyEventExtractor._nlp(text)

        events = []
        keywords_set = set(SpacyEventExtractor._keywords)
        for doc in text_doc.sents:
            # if there is no at least one keyword - we ignore that sentence
            if len(set([word.string.strip().lower() for word in doc]) & keywords_set) == 0:
                continue

            entity1 = SpacyEventExtractor._get_entity1(doc)
            if not entity1:
                continue
            verb = entity1.head
            entity2 = SpacyEventExtractor._get_entity2(verb)

            if SpacyEventExtractor._is_present_simple(verb) or \
                    SpacyEventExtractor._is_present_continuous(verb):
                continue

            entity1_string = SpacyEventExtractor._get_full_entity(entity1)
            entity2_string = SpacyEventExtractor._get_full_entity(entity2)

            entity1_string = SpacyEventExtractor._replace_we(replace_we, entity1_string)
            entity2_string = SpacyEventExtractor._replace_we(replace_we, entity2_string)

            entity1_string = SpacyEventExtractor._remove_extra_whitespaces(entity1_string)
            entity2_string = SpacyEventExtractor._remove_extra_whitespaces(entity2_string)

            # if there is no keywords in token and subj_string
            if len(set([word.strip().lower() for word in entity1_string.split()]) & keywords_set) + \
                    len(set(word.strip().lower() for word in entity2_string.split()) & keywords_set) == 0:
                continue

            if SpacyEventExtractor._have_pronouns(entity1_string) or \
                    SpacyEventExtractor._have_pronouns(entity2_string):
                continue

            # entity2 can be empty only in some special cases like: IDEA 2.0 released
            if verb.lemma_.lower() not in SpacyEventExtractor._important_actions and entity2_string == "":
                continue

            action_string = SpacyEventExtractor._get_action(verb)
            event = Event(entity1_string, entity2_string, action_string, str(doc))
            events.append(event)

            print(event)

        return events

def main():
    db_handler = DatabaseHandler()

    if 'clear_db' in sys.argv:
        db_handler.clear_db()

    if 'clear_events' in sys.argv:
        db_handler.cursor.execute("DELETE FROM event_sources")
        db_handler.cursor.execute("DELETE FROM events")
        db_handler.connection.commit()

    # simply downloading articles to database
    # todo: remove hardcoded number of articles
    for downloader in article_downloaders:
        try:
            for article in islice(downloader.get_articles(), 0, 1000):
                print(db_handler.add_article_or_get_id(article))
        except:
            import traceback
            print(traceback.format_exc())

    cnt = 1
    for article in db_handler.get_articles():
        print(article.url)
        events = SpacyEventExtractor.extract(article.summary, article.site_owner)
        # probably do not need to parse the whole text?
        # events += SpacyEventExtractor.extract(article.text, article.site_owner)
        events += SpacyEventExtractor.extract(article.header, article.site_owner)

        for event in events:
            print(db_handler.add_event_or_get_id(event, article), article.url)
        print(cnt)
        cnt += 1

if __name__ == '__main__':
    main()
