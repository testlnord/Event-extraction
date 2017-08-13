import logging as log
import sys
import os
from enum import Enum, IntEnum
from collections import Counter
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
    @classmethod
    def continue_tagging(cls, output_dir, choices, idefault_choice=-1, default_cursor_pos=0,
                         print_counts=True, data_printer=default_printer):

        with open(os.path.join(output_dir, 'processed.json')) as f:
            state, processed = json.load(f)
        assert all(itag < len(choices) for itag in processed if itag is not None)
        return cls(output_dir, choices, idefault_choice, default_cursor_pos,
                   state, processed,
                   print_counts, data_printer)

    # change order and names of arg-s
    def __init__(self, output_dir,
                 choices, idefault_choice=-1, default_cursor_pos=0,
                 state=0, processed=None,
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
        self._nchoices = len(self.choices)
        self.idefault_choice = idefault_choice
        self.default_cursor_pos = default_cursor_pos
        assert idefault_choice < self._nchoices and default_cursor_pos < self._nchoices, 'Index is out of bounds!'
        self._cursor = self.default_cursor_pos
        self.printer = data_printer
        self._print_counts = print_counts
        self.processed = processed
        self.state = state
        self._statusline = ''

    def run(self, records):
        # if isinstance(records, dict):
        #     todo: self.processed is a list of indices! not the choices itself
            # self.data, self.processed = zip(*records.items())
            # assert all(tag in self.choices for tag in self.processed)

        self.data = records
        self._ndata = len(records)
        if self.processed is None:
            self.processed = [None] * self._ndata
        assert len(self.processed) == self._ndata, "Provided 'processed' list must be of the same length as the data."
        assert all(hasattr(record, '__hash__') and hasattr(record, '__eq__') for record in records)\
            , 'Data records must be hashable.'

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

        while True:
            try:
                win.clear()

                # Initial choice
                if self.processed[self.state] is None:
                    self.processed[self.state] = self.idefault_choice
                self._counts = Counter(self.processed)

                # Print state and statusline
                win.addstr(y, x, '{}/{} {}'.format(self.state + 1, self._ndata, self._statusline))
                # Set styles for choices
                astyles = [curses.A_NORMAL] * len(self.choices)
                astyles[self._cursor] |= curses.A_UNDERLINE
                # Highlight actual choice
                ichoice = self.processed[self.state]
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
                self.printer(win, y + 3, x, self.data[self.state])

                win.refresh()
                # self._statusline = ''
            except curses.error:  # some meaningless error
                pass

            c = win.getch()
            if c == KeyMap.NEXT:
                self._cursor = self.default_cursor_pos
                self.state = min(self.state + 1, self._ndata - 1)
            elif c == KeyMap.PREV:
                self._cursor = self.default_cursor_pos
                self.state = max(self.state - 1, 0)
            elif c == KeyMap.SELECT:
                self.processed[self.state] = self._cursor
            elif c == KeyMap.NEXT_CHOICE:
                self._cursor = (self._cursor + 1) % self._nchoices
            elif c == KeyMap.PREV_CHOICE:
                self._cursor = (self._cursor - 1) % self._nchoices if self._cursor > 0 else self._nchoices - 1
            elif c == KeyMap.SAVE:
                self.save()
                self._statusline = 'saved {}'.format(self.state + 1)
            elif c == KeyMap.SAVE_AND_CLOSE:
                self.save()
                break

    def save_dict(self):
        tagged = dict(zip(self.data, self.processed))
        with open(os.path.join(self.output_dir, 'tagged_data.pck'), 'wb') as f:
            pickle.dump(tagged, f)

    def save(self):
        # Dump state (tags)
        with open(os.path.join(self.output_dir, 'processed.json'), 'w') as fp:
            json.dump((self.state, self.processed), fp)

        grouped = groupby(second, zip(self.data, self.processed))
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


# temporary
def from_partially_tagged(grouped, dataset, choices):
    from experiments.ontology.sub_ont import dbo
    len_author = len(grouped[dbo.author])
    len_computingPlatform = len(grouped[dbo.computingPlatform])
    __dir = '/home/user/datasets/dbpedia/rc/tagged/'
    tagger = ManualTagger.continue_tagging(__dir, choices, idefault_choice=1, data_printer=rrecord_printer)
    processed = tagger.processed[:min(num_cut, len_author)] + \
                tagger.processed[len_author:len_author + min(num_cut, len_computingPlatform)]
    state = len(processed) - 1
    processed += [None] * (len(dataset) - len(processed))
    return state, processed


def tag_crecords(output_dir, num_cut=None):
    from experiments.ontology.data import load_rc_data
    from experiments.ontology.symbols import RC_CLASSES_MAP

    choices = ['no', 'yes', None]
    sclasses = RC_CLASSES_MAP
    # data_dir = '/home/user/datasets/dbpedia/'
    data_dir = '/media/datasets/dbpedia/'
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

    # state, processed = from_partially_tagged(grouped, dataset, choices)
    # tagger = ManualTagger(output_dir, choices, idefault_choice=1, data_printer=rrecord_printer,
    tagger = ManualTagger.continue_tagging(output_dir, choices, idefault_choice=1, data_printer=rrecord_printer)
    tagger.run(dataset)


def raw(uri):
    return uri.rsplit('/', 1)[-1]


def main(output_dir):
    choices = ['no', 'yes', None]
    tagger = ManualTagger(output_dir, choices)
    # tagger = ManualTagger.continue_tagging(output_dir, choices, idefault_choice=-1)

    dummy_data = [
        'First string',
        'Second long string',
        'Third long long string',
        'Fourth string',
        'Fifth very.... ' + 'long ' * 40 + 'string!',
        'And so on...',
    ]

    tagger.run(dummy_data)
    print(tagger.processed)


if __name__ == "__main__":
    cdir = os.path.join(os.path.dirname(__file__), '..', '..')
    sys.path.append(cdir)
    from experiments.ontology.data_structs import RelationRecord

    num_cut = 500
    # output_dir = '/home/user/datasets/dbpedia/rc/golden{}/'.format(num_cut)
    output_dir = '/media/datasets/dbpedia/rc/golden{}/'.format(num_cut)
    tag_crecords(output_dir, num_cut)

    # output_dir = '/media/datasets/dbpedia/rc/test_tagger'
    # output_dir = '/home/user/datasets/dbpedia/rc/test_tagger'
    # main(output_dir)
