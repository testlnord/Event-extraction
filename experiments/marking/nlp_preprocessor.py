
class NLPPreprocessor:
    """Splits texts on sentences; drops short sentences (meausured in words)."""
    def __init__(self, nlp,  min_words_in_sentence=1):
        self.min_words = min_words_in_sentence
        self.nlp = nlp

    def sents(self, text: str):
        """Yields sentences from text"""
        # Replace apostrophe (&#8217;) to single quotation mark as spacy doesn't recognise it
        text = text.replace("’", "'")
        doc = self.nlp(text)
        for sent in doc.sents:
            if len(sent) >= self.min_words:
                yield sent

