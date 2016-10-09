class Event:
    def __init__(self, entity1, entity2, action, sentence, date=None, location=None, id=None):
        self.entity1 = entity1
        self.entity2 = entity2
        self.action = action
        self.sentence = sentence
        self.date = date
        self.location = location
        self.id = id

    def json(self):
        return self.__dict__

    def __repr__(self):
        return self.entity1 + '|' + self.action + '|' + self.entity2

    def __str__(self):
        adjust = 10
        return '\n'.join(['Sentence: '.rjust(adjust) + self.sentence.strip(),
                          'Object: '.rjust(adjust) + self.entity1.strip(),
                          'Action: '.rjust(adjust) + self.action.strip(),
                          'Subject: '.rjust(adjust) + self.entity2.strip()])
