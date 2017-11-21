from datetime import datetime
from configparser import ConfigParser
import os.path
import psycopg2
from psycopg2 import sql
from psycopg2.extras import NumericRange
from experiments.extraction.extraction import Extraction


class DBGate:
    config_filename = 'config.cfg'

    def __init__(self, nlp):
        self.nlp = nlp
        config = ConfigParser()
        root_dir = os.path.realpath(__file__).split('/')
        root_dir = '/' + os.path.join(*root_dir[:-3])  # todo: change accordingly when moving this from 'experiments' dir
        config_filename = os.path.join(root_dir, self.config_filename)
        config.read(config_filename)
        login = config['db']['login']
        password = config['db']['password']
        host = config['db']['host']
        dbname = config['db']['database']
        self.cnn = psycopg2.connect('dbname={} user={} password={} host={}'
                                    .format(dbname, login, password, host),
                                    # autocommit=True  # todo: autocommit?
                                    )
        # todo: test if connected

    # todo: add_source
    def add_source(self, s):
        with self.cnn:
            with self.cnn.cursor() as curs:
                pass

    def add_raw_text(self, t, source_id):
        with self.cnn:
            with self.cnn.cursor() as curs:
                curs.execute('''
                INSERT INTO raw_texts (source_uid, raw_text) 
                VALUES (%s, %s) 
                RETURNING text_uid''', (source_id, t))
                return curs.fetchone()[0]

    def add_span(self, span, text_pos, text_id):
        with self.cnn:
            with self.cnn.cursor() as curs:
                curs.execute('''
                INSERT INTO spans (text_uid, text_pos, span_text) 
                VALUES (%s, %s, %s) 
                RETURNING span_uid''', (text_id, text_pos, span.text))
                return curs.fetchone()[0]

    def add_extraction(self, e, parent_id, extractor_ver):
        with self.cnn:
            with self.cnn.cursor() as curs:
                args = [parent_id] + list(map(self._seq2range, [e.subject_span, e.relation_span, e.object_min_span, e.object_max_span])) + [extractor_ver]
                curs.execute('''
                INSERT INTO extractions (span_uid, subject, relation, object_min, object_max, extractor_ver)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING extraction_uid''', args)
                return curs.fetchone()[0]

    def add_entity(self, ent_span, ent_text, parent_id, extractor_ver):
        with self.cnn:
            with self.cnn.cursor() as curs:
                curs.execute('''
                INSERT INTO entities (extraction_uid, entity_span, entity_text)
                VALUES (%s, %s, %s, %s)
                RETURNING entity_uid''', (parent_id, self._seq2range(ent_span), ent_text, extractor_ver))
                return curs.fetchone()[0]

    def add_relation(self, rel_span, rel_text, parent_id, extractor_ver):
        with self.cnn:
            with self.cnn.cursor() as curs:
                curs.execute('''
                INSERT INTO relations (extraction_uid, relation_span, relation_text, extractor_ver)
                VALUES (%s, %s, %s, %s)
                RETURNING relation_uid''', (parent_id, self._seq2range(rel_span), rel_text, extractor_ver))
                return curs.fetchone()[0]

    # todo: check if table exists (i.e. attr is valid)
    def add_attribute(self, attr_span, attr_text, extraction_uid, attr_name, extractor_ver):
        with self.cnn:
            with self.cnn.cursor() as curs:
                table_name = 'attr_{}'.format(attr_name)
                q = sql.SQL('''
                INSERT INTO {} (extraction_uid, attr_span, attr_text, extractor_ver)
                VALUES (%s, %s, %s, %s) RETURNING attr_uid'''.format(sql.Identifier(table_name)))
                curs.execute(q, (extraction_uid, self._seq2range(attr_span), attr_text, extractor_ver))
                return curs.fetchone()[0]

    # todo: add select by extraction_ver
    # todo: add select by source_type

    # todo:
    def get_sources(self):
        with self.cnn.cursor() as curs:
            curs.execute('''SELECT * FROM sources''')
            return curs.fetchall()

    def get_raw_texts(self, dt_from=datetime.min, dt_to=datetime.max):
        with self.cnn.cursor() as curs:
            curs.execute('''
            SELECT text_uid, raw_text
            FROM raw_texts
            WHERE dt_extracted >= (%s) AND dt_extracted < (%s)''', [dt_from, dt_to])
            for text_uid, raw_text in curs:
                yield text_uid, self.nlp(raw_text)

    def get_spans(self, dt_from=datetime.min, dt_to=datetime.max):
        with self.cnn.cursor() as curs:
            curs.execute('''
            SELECT span_uid, span_text
            FROM texts_spans
            WHERE dt_extracted >= (%s) AND dt_extracted < (%s)''', [dt_from, dt_to])
            for span_uid, span_text in curs:
                yield span_uid, self.nlp(span_text)

    def get_extractions(self, dt_from=datetime.min, dt_to=datetime.max):
        with self.cnn.cursor() as curs:
            curs.execute('''
            SELECT extraction_uid, subject, relation, object_min, object_max, span_text, span_uid
            FROM spans_extractions
            WHERE dt_extracted >= (%s) AND dt_extracted < (%s)''', [dt_from, dt_to])
            for things in curs:
                subj_span = self._range2tuple(things[1])
                rel_span = self._range2tuple(things[2])
                objmin_span = self._range2tuple(things[3])
                objmax_span = self._range2tuple(things[4])
                span = self.nlp(things[5])
                yield things[0], Extraction(span, subj_span, rel_span, objmin_span, objmax_span)

    def _range2tuple(self, range):
        lower = range.lower + int(not range.lower_inc)
        upper = range.upper + int(range.upper_inc)
        return [lower, upper]

    def _seq2range(self, seq):
        """Assuming seq[:2] is closed-open range, return psycopg2-friendly range type."""
        return NumericRange(lower=seq[0], upper=seq[1])

