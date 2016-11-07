import logging as log
from event import Event


# todo: make check of offsets and actual extractions equality
class EventOffsets(Event):
    """Event with information about offsets of its' entities in the text. Offsets measured in symbols."""
    def __init__(self, entity1_offs, entity2_offs, action_offs, sentence,
                 entity1=None, entity2=None, action=None):
        self.sentence = sentence
        self.entity1_offsets = entity1_offs
        self.action_offsets = action_offs
        self.entity2_offsets = entity2_offs

        entity1 = self._verify(entity1_offs, entity1)
        entity2 = self._verify(entity2_offs, entity2)
        action = self._verify(action_offs, action)
        super().__init__(entity1=entity1, entity2=entity2, action=action, sentence=sentence)

    def _substring(self, offsets):
        sent = self.sentence
        s = [sent[o[0]:o[1]] for o in offsets]
        return ' '.join(s)

    def _verify(self, obj_offsets, actual_obj=None):
        """Check that obj_offsets are specifying exactly actual_obj.
        If not, then modify offsets to match actual_obj.
        If actual_obj is not known then just extract obj using known offsets."""

        offsetted = self._substring(obj_offsets)
        if actual_obj:
            if abs(len(offsetted) - len(actual_obj)) > 1:
                log.debug('EventOffsets: offsets not matching actual_obj: '
                         'offsets={}, offsetted="{}", actual_obj="{}"'.format(obj_offsets, offsetted, actual_obj))
                # todo:
                # find right offsets using real sentence, beggining of offsets(it should be true) and actual_obj
                # then modify offsets and return actual_obj
            return actual_obj
        else:
            return offsetted

    @property
    def offsets(self):
        return self.entity1_offsets + self.action_offsets + self.entity2_offsets

