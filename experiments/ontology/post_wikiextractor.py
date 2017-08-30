#!/bin/python3

import logging as log
import daiquiri
import os
import json
from collections import defaultdict
import urllib
from bs4 import BeautifulSoup


def test_article(art):
    bads = 0
    for res, _l in art['links'].items():
        for _tbeg, _tend, htext in _l:
            __cc = art['text'][_tbeg:_tend]
            so = int(__cc != htext)
            bads += so
            if so == 1:
                print('{}: <{}> ? <{}>'.format(so, htext, __cc))
    return bads


def extract_hrefs(text):
    """Return dict of links to links' texts and text with removed href attributes. The dict if None."""
    d = defaultdict(list)
    clean_text = ''

    abeg = '<a href="'
    aend = '</a>'
    prev_end = 0
    pos = 0
    chars_removed_so_far = 0
    unquote = urllib.parse.unquote
    htext = ''
    while True:
        beg = text.find(abeg, pos)
        if beg > 0:
            hbeg = beg + len(abeg)
            pos = hbeg + 1
            hend = text.find('">', pos)
            if hend > 0:
                href = text[hbeg:hend]
                pos = hend + 1
                tbeg = hend + 2
                tend = text.find(aend, pos)
                if tend > 0:
                    htext = text[tbeg:tend]
                    end = tend + len(aend)
                    pos = end + 1
                    #log.INFO(text[beg:end])
                    #log.INFO(href, ':', htext)
                    
                    #if not href.startswith('http'):
                    resource = unquote(href).replace(' ', '_')  # transform to dbpedia format
                    clean_text += text[prev_end:beg] + htext
                    prev_end = end
                    chars_removed_so_far += (tbeg - beg)
                    _tbeg = tbeg - chars_removed_so_far
                    _tend = tend - chars_removed_so_far
                    chars_removed_so_far += (end - tend)
                    d[resource].append((_tbeg, _tend, htext))
                    if htext != clean_text[_tbeg:_tend]:
                        log.warning('extracted href text and text in cleaned text are not the same! {} != {}'.format(htext, clean_text[_tbeg:_tend]))
            continue
        break
    # TODO: when the entity is in the end of the text it is not included (seems like that)
    clean_text += text[prev_end:]
    return dict(d), clean_text


def extract_hrefs0(text):
    """Return dict of links to links' texts and text with removed href attributes. The dict if None."""
    # TODO: keep offsets from original text? keep offsets with respect to cleaned text?
    d = defaultdict(list)
    unquote = urllib.parse.unquote
    soup = BeautifulSoup(text)
    links = soup.findAll('a')
    for link in links:
        href = link.get('href')
        # Check that it is internal wikipedia link
        if not href.startswith('http'):
            resource = unquote(href).replace(' ', '_')  # transform to dbpedia format
            d[resource].append(link.text)
        link.replace_with(link.text)
    return  dict(d), soup.text


def decouple(inpdir, outdir, visited=set()):
    """Returns mapping of articles ids to article titles"""
    id_names = dict()
    for root, subdirs, files in os.walk(inpdir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fpath not in visited:
                visited.add(fpath)
                with open(fpath) as f:
                    log.info(fpath)  # for logging
                    for line in f.readlines():
                        jart = json.loads(line)
                        _id = jart['id']
                        name = _id
                        id_names[int(_id)] = jart['title']
                        links, clean_text = extract_hrefs(jart['text'])
                        jart['links'] = links
                        jart['text'] = clean_text

                        #bad_counts = test_article(jart)
                        #if bad_counts > 0:
                        #    log.warning('{} bad extractions in article "{}" (id={})!'.format(bad_counts, jart['title'], _id))

                        with open(os.path.join(outdir, name), 'w') as fout:
                            json.dump(jart, fout)
    return id_names


def main(inpdir, outdir, visited_file=None):
    visited = set() if visited_file is None else set(json.load(open(visited_file)))
    try:
        id_names = decouple(inpdir, outdir, visited)
        with open(os.path.join(outdir, 'id_names.json'), 'w') as fnames:
            json.dump(id_names, fnames)
    except:
        dumped = json.dumps(list(visited))
        print('visited filepaths: {}'.format(dumped))
        with open(os.path.join(outdir, 'visited_filepaths.json'), 'w') as f:
                f.write(dumped)
        raise


if __name__ == "__main__":
    daiquiri.setup(level=log.INFO)
    log.warning('test warning log')
    #import plac
    #plac.call(main)
    inpdir = '/home/user/datasets/dbpedia/articles_tmp/'
    outdir = '/home/user/datasets/dbpedia/articles4/'
    main(inpdir, outdir)


