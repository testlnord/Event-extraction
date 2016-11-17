from itertools import chain
from experiments.marking.preprocessor import Preprocessor


# todo: it is data fetcher and preprocessor. maybe divide somehow
class NERPreprocessor:
    def objects(self):
        for sent, tags in chain(self._wikigold_conll(), self._wikiner_wp3()):
            yield sent, self._check_tags(tags)

    def _wikigold_conll(self, path='/media/Documents/datasets/ner/wikigold.conll.txt'):
        with open(path) as f:
            sent = []
            tags = []
            for line in f:
                # print(line.strip())
                if len(line) > 1:
                    token, ner_tag = line.strip().split(' ', maxsplit=2)
                    bio2_tag = ner_tag[0]
                    sent.append(token)
                    tags.append(bio2_tag)
                    assert bio2_tag in ('O', 'I', 'B')
                else:
                    yield sent, tags
                    sent.clear()
                    tags.clear()

    def _wikiner_wp3(self, path='/media/Documents/datasets/ner/aij-wikiner-en-wp3'):
        with open(path) as f:
            for line in f:
                sent = []
                tags = []
                for token_tag in line.strip().split(' '):
                    if len(token_tag) > 1:
                        token, _, ner_tag = token_tag.split('|')
                        bio2_tag = ner_tag[0]
                        sent.append(token)
                        tags.append(bio2_tag)
                        assert bio2_tag in ('O', 'I', 'B')
                yield sent, tags

    def _check_tags(self, tags):
        """Change 'I' tags to 'B' tags where necessary (B is for the Beginning of NE)"""
        strtags = 'O' + ''.join(tags)
        strtags = strtags.replace('OI', 'OB')
        newtags = [c for c in strtags][1:]
        assert len(newtags) == len(tags)
        return newtags


if __name__ == "__main__":
    preprocessor = NERPreprocessor()
    for sent, tags in preprocessor.objects():
        print(' '.join(sent))
        print(' '.join(tags))
        pass
