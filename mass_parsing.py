import json, os

import wsd_config
from gibber import parsing
from gibber.annot_corp_loader import load_wn3_corpus

from lxml import etree
#
# Establish desired lexids/words from Wordnet to have their descripitions parsed.
#
desired_lexids = set()
desired_words = set()

if wsd_config.mass_parsing_mode == 'corpus':
    _, tagged_words = load_wn3_corpus(wsd_config.annot_sentences_path)
    desired_words = set([lemma for (lemma, tag) in tagged_words])
else:
    with open(wsd_config.mass_parsing_lexids_file) as lexids_file:
        desired_lexids = set(json.load(lexids_file))

#
# Collect descriptions from Wordnet.
#
wordnet_xml = etree.parse(wsd_config.pl_wordnet_path)

wn_wordid_descs = dict()
for lex_unit in wordnet_xml.iterfind('lexical-unit'):
    if lex_unit.get('id') in desired_lexids or lex_unit.get('name').lower() in desired_words:
        useful_desc = pl_parsing.extract_samples(lex_unit.get('desc')).strip()
        if len(useful_desc) > 0:
                wn_wordid_descs[lex_unit.get('id')] = useful_desc

#
# Write the parsing outcomes.
#
os.makedirs(wsd_config.mass_parsing_path, exist_ok=True)

ordered_lexids = list(wn_wordid_descs.keys())
print('{} lexical ids staged for description parsing.'.format(len(ordered_lexids)))
with open(wsd_config.mass_parsing_path+'/All_requested', 'w+') as out:
    print(json.dumps(ordered_lexids), file=out)

for li_n, lexid in enumerate(ordered_lexids):
    print('Parsing {}/{}: {}'.format(li_n+1, len(ordered_lexids), lexid))
    parsed_sents = pl_parsing.parse_sentences(wn_wordid_descs[lexid])
    with open(wsd_config.mass_parsing_path+'/'+lexid+'.txt', 'w+') as out:
        print(json.dumps(parsed_sents), file=out)

print() # newline
