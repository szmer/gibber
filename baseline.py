import os, csv, re
from itertools import chain

from operator import itemgetter
from lxml import etree

from wsd_config import nkjp_index_path, mode, skladnica_sections_index_path, skladnica_path, annot_sentences_path, baseline_output_prefix, full_diagnostics, diagnostics_when_23_fails, pl_wordnet_path, baseline
from gibber_wsd import add_word_neighborhoods, fill_sample, predict_sense, sense_match, random_prediction, first_variant_prediction
import gibber_wsd
from annot_corp_loader import load_skladnica_wn2, load_wn3_corpus

print('mode: {}\nNKJP: {}\nWordnet: {}'.format(mode, nkjp_index_path, pl_wordnet_path))
print('baseline: {}'.format(baseline))

### Get Składnica sentences

# NOTE this also contains empty lists where the sents were actually not annotated.
sents = [] # pairs: (word lemma, lexical unit id [or None])
words = set() # all unique words that are present

if mode == 'wordnet2_annot':
    sents, words = load_skladnica_wn2(skladnica_path, skladnica_sections_index_path)

if mode == 'wordnet3_annot':
    sents, words = load_wn3_corpus(annot_sentences_path)

#
# Load all the necessary wordnet data.
#
print('Loading Wordnet data.')
add_word_neighborhoods(words)

# ## Tests.
num_all = 0
num_good = 0

if baseline_output_prefix:
    out = open(baseline_output_prefix, 'w+')

for sent in sents:
    for (tid, token_data) in enumerate(sent):
        lemma, true_sense, tag = token_data[0], token_data[1], token_data[2]
        if true_sense is None:
            if baseline_output_prefix is not None:
                print('{},{},{},{}'.format(lemma, tag, '<<<', '<<<'), file=out)
            continue

        num_all += 1

        try:
            if baseline == 'random':
                if full_diagnostics:
                    decision = random_prediction(lemma, tag, verbose=True)
                else:
                    decision = random_prediction(lemma, tag)
            elif baseline == 'first-variant':
                if full_diagnostics:
                    decision = first_variant_prediction(lemma, tag, verbose=True)
                else:
                    decision = first_variant_prediction(lemma, tag)
            else:
                raise ValueError('baseline must be specified as random or first-variant, provided as {}'.format(baseline))
        except LookupError as err:
            print(err)
            continue

        if decision is not None and sense_match(decision, true_sense):
            num_good += 1
        if baseline_output_prefix is not None:
            if decision is not None:
                sense_data = decision.split('_')
                print('{},{},{},{}'.format(lemma, tag, sense_data[2], sense_data[0]), file=out)
            else:
                print('{},{},{},{}'.format(lemma, tag, '?', '?'), file=out)

print('{} cases, {} predicted correctly, {} accuracy'.format(num_all, num_good, num_good/num_all))
