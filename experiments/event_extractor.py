from event import Event

class EventExtractor:
    def extract_event(self, sentence, category):
        """
        Extracts relations with following structure: (NE chunk, <linking word(s)>, NE chunk)
        where NE is Named Entity with related words
        and <linking word(s)> is one or more words that syntactically connect two NE chunks
        """

        entity1 = None
        entity2 = None
        action = None
        date = None
        location = None

        # todo: entity1, entity2 and action extraction heuristics
        for token in sentence:
            pass

        event = Event(entity1=entity1, entity2=entity2, action=action, sentence=sentence,
                      date=date, location=location)
        return event

    def shortest_path(self, span1, span2):
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

    def numbers_heuristic(self):
        """
        Extract version/product numbers
        (e.g. Windows -> Windows 10)"""
        pass

    def expand_heuristic(self, token):
        """
        Extract full entity having part of the entity
        (e.g. Windows Insider -> the Windows Insider program)
        """
        pass

