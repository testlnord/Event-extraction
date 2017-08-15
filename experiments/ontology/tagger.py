import logging as log
import random
import sys
import os
from enum import Enum, IntEnum
from collections import Counter, OrderedDict, namedtuple
import pickle
import json
import curses
from curses import wrapper

from cytoolz import groupby, first, second, sliding_window


class KeyMap(IntEnum):
    NEXT = curses.KEY_DOWN
    PREV = curses.KEY_UP
    SELECT = ord(' ')
    NEXT_CHOICE = curses.KEY_RIGHT
    PREV_CHOICE = curses.KEY_LEFT
    SAVE = 10  # Enter
    SAVE_AND_CLOSE = 27  # and save


def default_printer(stdscr, y, x, record):
    stdscr.addstr(y, x, str(record))


class ManualTagger:
    _annotated_filename = 'annotated_data.pck'

    def __init__(self, output_dir,
                 choices, idefault_choice=-1, default_cursor_pos=0,
                 print_counts=True, data_printer=default_printer):
        """

        :param output_dir:
        :param choices: tagging choices
        :param idefault_choice: index of choice from @choices selected by default
        :param default_cursor_pos: default position of the cursor
        :param state: record in dataset to start from
        :param processed: list of indexes of tags for entities (default None - no tags provided)
        :param print_counts: print counts of tagged records or not
        :param data_printer: function printing data, taking as arguments (curses.screen, y, x, data_to_print)
        """
        self.output_dir = output_dir
        self.choices = choices
        assert all(choice is not None for choice in self.choices)
        self._nchoices = len(self.choices)
        self.idefault_choice = idefault_choice
        self.default_cursor_pos = default_cursor_pos
        assert idefault_choice < self._nchoices and default_cursor_pos < self._nchoices, 'Index is out of bounds!'
        self._cursor = self.default_cursor_pos
        self.printer = data_printer
        self._print_counts = print_counts

    def continue_tagging(self):
        with open(os.path.join(self.output_dir, self._annotated_filename), 'rb') as f:
            annotated = pickle.load(f)
            # assert isinstance(annotated, OrderedDict)
        ch_map = dict((choice, i) for i, choice in enumerate(self.choices))
        records, annotations = zip(*annotated.items())
        _processed = [ch_map[annotation] if annotation is not None else None for annotation in annotations]
        _pos = 0
        max_pos = len(_processed)-1
        while _pos < max_pos and _processed[_pos] is not None:
            _pos += 1  # choosing first untagged record
        self.run(records, _processed, _pos)

    def run(self, records, processed=None, state=0):
        assert all(hasattr(record, '__hash__') and hasattr(record, '__eq__') for record in records) \
            , 'Data records must be hashable.'
        self.data = records
        self._ndata = len(records)
        self._processed = [None] * self._ndata if not processed else processed
        self._pos = state
        assert len(self._processed) == self._ndata
        assert self._pos < self._ndata

        try:
            wrapper(self._run)
        except KeyboardInterrupt:
            log.info('Received KeyboardInterrupt; exiting.')
        finally:
            curses.endwin()

    def _run(self, stdscr):
        curses.curs_set(False)
        # wy = 3
        # wx = 1
        # win = stdscr.derwin(wy, wx)
        win = stdscr
        y = 2
        x = 1
        self._schoices = list(map(str, self.choices))
        # _pad = max(map(len, self._schoices)) + 2
        self._statusline = ''

        while True:
            try:
                win.clear()

                # Initial choice
                if self._processed[self._pos] is None:
                    self._processed[self._pos] = self.idefault_choice
                self._counts = Counter(self._processed)

                # Print state and statusline
                win.addstr(y, x, '{}/{} {}'.format(self._pos + 1, self._ndata, self._statusline))
                # Set styles for choices
                astyles = [curses.A_NORMAL] * len(self.choices)
                astyles[self._cursor] |= curses.A_UNDERLINE
                # Highlight actual choice
                ichoice = self._processed[self._pos]
                if ichoice is not None:
                    astyles[ichoice] |= curses.A_BOLD
                # Print choices
                offset = 0
                for i, (sch, astyle) in enumerate(zip(self._schoices, astyles)):
                    win.addstr(y + 1, x + offset, sch, astyle)  # moving cursor
                    offset += len(sch)
                    if self._print_counts:
                        scount = ' ({})'.format(self._counts[i])
                        win.addstr(y + 1, x + offset, scount)
                        offset += len(scount)
                    offset += 2
                # Print record
                self.printer(win, y + 3, x, self.data[self._pos])

                win.refresh()
            except curses.error:  # some meaningless error
                pass

            c = win.getch()
            if c == KeyMap.NEXT:
                self._cursor = self.default_cursor_pos
                self._pos = min(self._pos + 1, self._ndata - 1)
            elif c == KeyMap.PREV:
                self._cursor = self.default_cursor_pos
                self._pos = max(self._pos - 1, 0)
            elif c == KeyMap.SELECT:
                self._processed[self._pos] = self._cursor
            elif c == KeyMap.NEXT_CHOICE:
                self._cursor = (self._cursor + 1) % self._nchoices
            elif c == KeyMap.PREV_CHOICE:
                self._cursor = (self._cursor - 1) % self._nchoices if self._cursor > 0 else self._nchoices - 1
            elif c == KeyMap.SAVE:
                self.save_dict()
                self._statusline = 'saved {}'.format(self._pos + 1)
            elif c == KeyMap.SAVE_AND_CLOSE:
                self.save_dict()
                break

    def save_dict(self):
        """Save mapping of data to annotations"""
        choices = [self.choices[i] if i is not None else None for i in self._processed]
        tagged = OrderedDict(zip(self.data, choices))
        with open(os.path.join(self.output_dir, self._annotated_filename), 'wb') as f:
            pickle.dump(tagged, f)

    def save_to_files(self):
        """Save data with the same annotations to the same files"""
        grouped = groupby(second, zip(self.data, self._processed))
        for ichoice, choice in enumerate(self.choices):
            # Opening files anyway to empty them
            with open(os.path.join(self.output_dir, str(choice) + '.pck'), 'wb') as f:
                if ichoice in grouped:
                    for record in grouped[ichoice]:
                        pickle.dump(record, f)
        # Saving untagged data separately
        with open(os.path.join(self.output_dir, '__untagged__.pck'), 'wb') as f:
            if None in grouped:
                for record in grouped[None]:
                    pickle.dump(record, f)


def rrecord_printer(win, y, x, record):
    # triple_str = '<{}> <{}> <{}>'.format(*map(raw, record.triple))
    # stdscr.addstr(y+0, x, triple_str)
    win.addstr(y + 0, x, 'relation: <{}>'.format(raw(record.relation)))
    win.addstr(y + 1, x, ' subject: <{}>'.format(raw(record.subject)))
    win.addstr(y + 2, x, '  object: <{}>'.format(raw(record.object)))

    ent_style = curses.A_BOLD
    text = record.context
    a1 = record.s_startr
    b1 = record.s_endr
    a2 = record.o_startr
    b2 = record.o_endr
    if a1 > a2:
        a1, b1, a2, b2 = a2, b2, a1, b1  # swap
    win.move(y + 4, x)
    win.addstr(text[:a1])
    win.addstr(text[a1:b1], ent_style)
    win.addstr(text[b1:a2])
    win.addstr(text[a2:b2], ent_style)
    win.addstr(text[b2:])


RecordAnnotation = namedtuple('RecordAnnotation', ['positive', 'negative', 'none'])
default_annotations = RecordAnnotation('yes', 'no', 'None')


def load_golden_data(allowed_classes, rc_dir, shuffle=True, annotations=default_annotations):
    assert isinstance(annotations, RecordAnnotation)

    filename = 'annotated_data.pck'
    with open(os.path.join(rc_dir, filename), 'rb') as f:
        d = pickle.load(f)

    records = []
    _classes = set(allowed_classes)
    for record, annotation in d.items():
        if annotation == annotations.negative:
            record.relation = None
            records.append(record)
        elif annotation == annotations.none:
            continue
        if str(record.relation) in _classes:
            records.append(record)
    if shuffle: random.shuffle(records)
    return records


def tag_crecords(output_dir, num_cut=None):
    from experiments.ontology.data import load_rc_data
    from experiments.ontology.symbols import RC_CLASSES_MAP, RC_CLASSES_MAP_MORE

    sclasses = RC_CLASSES_MAP_MORE
    data_dir = '/home/user/datasets/dbpedia/'
    # data_dir = '/media/datasets/dbpedia/'
    rc_out = os.path.join(data_dir, 'rc', 'rrecords.v2.filtered.pck')
    rc0_out = os.path.join(data_dir, 'rc', 'rrecords.v2.negative.pck')
    dataset = load_rc_data(sclasses, rc_file=rc_out, rc_neg_file=rc0_out, neg_ratio=0., shuffle=False)

    def get_rel(rr): return rr.relation

    sorted_dataset = list(sorted(dataset, key=get_rel))
    grouped = groupby(get_rel, sorted_dataset)
    # Tag at most num_cut records
    dataset = [grouped[key][:num_cut] for key in sorted(grouped.keys())]
    print(list(map(len, dataset)))
    dataset = sum(dataset, list())

    output_dir = os.path.join(output_dir, 'more')
    tagger = ManualTagger(output_dir, choices=default_annotations,
                          idefault_choice=0, default_cursor_pos=1, data_printer=rrecord_printer)
    tagger.run(dataset)
    # tagger.continue_tagging()

    # transition from previous tagger version
    # with open(os.path.join(output_dir, 'processed.json')) as f:
    #     state, processed = json.load(f)
    # for i, item in enumerate(processed):
    #     if item is None:
    #         processed[i] = 'None'
    # tagger.run(dataset, processed, state)


def raw(uri):
    return uri.rsplit('/', 1)[-1]


def main(output_dir):
    choices = ['no', 'yes', 'None']
    tagger = ManualTagger(output_dir, choices)

    dummy_data = [
        'First string',
        'Second long string',
        'Third long long string',
        'Fourth string',
        'Fifth very.... ' + 'long ' * 40 + 'string!',
        'And so on...',
    ]

    tagger.run(dummy_data)
    print(tagger._processed)


if __name__ == "__main__":
    cdir = os.path.join(os.path.dirname(__file__), '..', '..')
    sys.path.append(cdir)
    from experiments.ontology.data_structs import RelationRecord  # for unpickling

    num_cut = 500
    golden_dir = '/home/user/datasets/dbpedia/rc/golden{}/'.format(num_cut)
    tag_crecords(golden_dir, num_cut)

    # output_dir = '/home/user/datasets/dbpedia/rc/test_tagger'
    # main(output_dir)
