import json, os

import wsd_config
from gibber import parsing
from gibber.annot_corp_loader import load_skladnica_wn2, load_wn3_corpus, load_kpwr_corpus

from lxml import etree
#
# Establish desired identifiers/words from Wordnet to have their descripitions parsed.
#
desired_identifiers = set()
desired_words = set()

if wsd_config.mass_parsing_mode == 'corpus':
    if wsd_config.mode == 'wordnet3_annot':
        _, tagged_words = load_wn3_corpus(wsd_config.annot_sentences_path)
    elif wsd_config.mode == 'wordnet2_annot':
        _, tagged_words = load_skladnica_wn2(wsd_config.skladnica_path, wsd_config.skladnica_sections_index_path)
    elif wsd_config.mode == 'kpwr_annot':
        _, tagged_words = load_kpwr_corpus(wsd_config.kpwr_path)
    else:
        raise NotImplementedError('unknown mode {}'.format(wsd_config.mode))
    desired_words = set([lemma for (lemma, tag) in tagged_words])
else:
    with open(wsd_config.mass_parsing_identifiers_file, encoding='utf-8') as identifiers_file:
        desired_identifiers = set(json.load(identifiers_file))

#
# Collect descriptions from Wordnet.
#
wordnet_xml = etree.parse(wsd_config.pl_wordnet_path)

wn_wordid_descs = dict()
for lex_unit in wordnet_xml.iterfind('lexical-unit'):
    identifier = '{}_{}_{}'.format(lex_unit.get('id'), lex_unit.get('name').lower(), lex_unit.get('variant'))
    if identifier in desired_identifiers or lex_unit.get('name').lower() in desired_words:
        useful_desc = lex_unit.get('desc').strip()
        # (earlier parsing.extract_samples(lex_unit.get('desc')).strip())
        if len(useful_desc) > 0:
            wn_wordid_descs[identifier] = useful_desc

#
# Write the parsing outcomes.
#
os.makedirs(wsd_config.mass_parsing_path, exist_ok=True)

ordered_identifiers = list(wn_wordid_descs.keys())
print('{} senses staged for description parsing.'.format(len(ordered_identifiers)))
with open(wsd_config.mass_parsing_path+'/All_requested', 'w+', encoding='utf-8') as out:
    print(json.dumps(ordered_identifiers), file=out)

for li_n, identifier in enumerate(ordered_identifiers):
    print('Parsing {}/{}: {}'.format(li_n+1, len(ordered_identifiers), identifier))
    parsed_sents = parsing.parse_sentences(wn_wordid_descs[identifier])
    with open(wsd_config.mass_parsing_path+'/'+identifier+'.txt', 'w+', encoding='utf-8') as out:
        print(json.dumps(parsed_sents), file=out)

print() # newline
