import os, csv, re
from itertools import chain

from wsd_config import nkjp_index_path, mode, skladnica_sections_index_path, skladnica_path, annot_sentences_path, output_prefix, full_diagnostics, diagnostics_when_23_fails, pl_wordnet_path, use_descriptions 
from gibber_wsd import add_word_neighborhoods, predict_sense, sense_match
import gibber_wsd
from annot_corp_loader import load_skladnica_wn2, load_wn3_corpus

print('mode: {}\nNKJP: {}\nWordnet: {}'.format(mode, nkjp_index_path, pl_wordnet_path))

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

# ## Tests

##
## Collect model decisions for all subsets of relations, counting correct decisions.
##

## numbers indicate the relation group,
## A - words where at least one sense has known neighbors,
## B - words where all senses have known neighbors,
## uq - no word repetitions
## "Good" values will be divided by N to get accuracy.
words_checked_a = set() # for determining uniqueness
# If there is one connected sense in the 1st relation set, there is in all the other (which include this set).
# The b (full coverage) checked word sets are such that if some word is noth in the 1st set, it can be in some
# of the subsequent ones.
words_checked_b1 = set()
words_checked_b2 = set()
words_checked_b3 = set()
words_checked_b4 = set()
words_checked_b5 = set() # for descriptions

words_checked_rest = set() # for calculating the total performance
num_all = 0 # number of all occurences where we have some true sense provided

num_a1 = 0
num_a2 = 0
num_a3 = 0
num_a4 = 0
num_a5 = 0
num_b1 = 0
num_b2 = 0
num_b3 = 0
num_b4 = 0
num_b5 = 0
# also, len(words_checked)
good_a1 = 0.0
good_b1 = 0.0
good_a2 = 0.0
good_b2 = 0.0
good_a3 = 0.0
good_b3 = 0.0
good_a4 = 0.0
good_b4 = 0.0
good_a5 = 0.0
good_b5 = 0.0
good_a1_uq = 0.0
good_b1_uq = 0.0
good_a2_uq = 0.0
good_b2_uq = 0.0
good_a3_uq = 0.0
good_b3_uq = 0.0
good_a4_uq = 0.0
good_b4_uq = 0.0
good_a5_uq = 0.0
good_b5_uq = 0.0

##
## Collect model decisions for all subsets of relations, counting correct decisions.
##

if output_prefix is not None:
    out1 = open(output_prefix+'1', 'w+')
    out2 = open(output_prefix+'2', 'w+')
    out3 = open(output_prefix+'3', 'w+')
    out4 = open(output_prefix+'4', 'w+')
    if use_descriptions:
        out5 = open(output_prefix+'5', 'w+')

print('Performing the test on sense-annotated sentences.')
for (sid, sent) in enumerate(sents):
    print('Testing sentence', sid+1)
    for (tid, token_data) in enumerate(sent):
        lemma, true_sense, tag = token_data[0], token_data[1], token_data[2]
        if true_sense is None:
            if output_prefix is not None:
                print('{},{},{},{}'.format(lemma, tag, '<<<', '<<<'), file=out1)
                print('{},{},{},{}'.format(lemma, tag, '<<<', '<<<'), file=out2)
                print('{},{},{},{}'.format(lemma, tag, '<<<', '<<<'), file=out3)
                print('{},{},{},{}'.format(lemma, tag, '<<<', '<<<'), file=out4)
                if use_descriptions:
                    print('{},{},{},{}'.format(lemma, tag, '<<<', '<<<'), file=out5)
            continue

        num_all += 1

        try:
            if use_descriptions:
                decision1, decision2, decision3, decision4, decision5 = predict_sense(lemma, tag,
                    [tok_info[0] for tok_info in sent], # give only lemmas
                    tid, verbose=full_diagnostics)
            else:
                decision1, decision2, decision3, decision4 = predict_sense(lemma, tag,
                    [tok_info[0] for tok_info in sent], # give only lemmas
                    tid, verbose=full_diagnostics)
        except LookupError as err:
            print(err)
            words_checked_rest.add(lemma)
            continue

        if [None, None, None, None] == [decision1, decision2, decision3, decision4]:
            words_checked_rest.add(lemma)
            continue

        new_word = False
        if not lemma in words_checked_a:
            words_checked_a.add(lemma)
            new_word = True

        good1 = False # indicate if the first decision was good to show diagnostics when 3 or 3 fails
        good2 = False
        good3 = False

        if decision1 is not None:
            sense_id, prob, senses_count, considered_sense_count = decision1
            if output_prefix is not None:
                sense_data = sense_id.split('_')
                print('{},{},{},{}'.format(lemma, tag, sense_data[2], sense_data[0]), file=out1)
            
            # Increase occurence counts.
            num_a1 += 1
            if senses_count == considered_sense_count:
                num_b1 += 1

            # Correct?
            if sense_match(sense_id, true_sense):
                good_a1 += 1
                good1 = True
                if senses_count == considered_sense_count:
                    good_b1 += 1

            # Unique word indication.
            if new_word:
                if senses_count == considered_sense_count:
                    words_checked_b1.add(lemma) # marked as checked with all info on all senses

                if sense_match(sense_id, true_sense):
                    good_a1_uq += 1
                    if senses_count == considered_sense_count:
                        good_b1_uq += 1
        elif output_prefix is not None:
            print('{},{},{},{}'.format(lemma, tag, '?', '?'), file=out1)

        if decision2 is not None or decision1 is not None:
            if decision2 is None:
                sense_id, prob, senses_count, considered_sense_count = decision1
                if good1:
                    good2 = True
            else:
                sense_id, prob, senses_count, considered_sense_count = decision2

            if output_prefix is not None:
                sense_data = sense_id.split('_')
                print('{},{},{},{}'.format(lemma, tag, sense_data[2], sense_data[0]), file=out2)
            
            # Increase occurence counts.
            num_a2 += 1
            if senses_count == considered_sense_count:
                num_b2 += 1

            # Correct?
            if sense_match(sense_id, true_sense):
                good_a2 += 1
                good2 = True
                if senses_count == considered_sense_count:
                    good_b2 += 1

            # Unique word indication.
            if new_word:
                if senses_count == considered_sense_count and not lemma in words_checked_b1:
                    words_checked_b2.add(lemma) # marked as checked with all info on all senses

                if sense_match(sense_id, true_sense):
                    good_a2_uq += 1
                    if senses_count == considered_sense_count:
                        good_b2_uq += 1
        elif output_prefix is not None:
            print('{},{},{},{}'.format(lemma, tag, '?', '?'), file=out2)

        if decision3 is not None or decision1 is not None:
            if decision3 is None:
                sense_id, prob, senses_count, considered_sense_count = decision1
                if good1:
                    good3 = True
            else:
                sense_id, prob, senses_count, considered_sense_count = decision3
            
            if output_prefix is not None:
                sense_data = sense_id.split('_')
                print('{},{},{},{}'.format(lemma, tag, sense_data[2], sense_data[0]), file=out3)

            # Increase occurence counts.
            num_a3 += 1
            if senses_count == considered_sense_count:
                num_b3 += 1

            # Correct?
            if sense_match(sense_id, true_sense):
                good_a3 += 1
                good3 = True
                if senses_count == considered_sense_count:
                    good_b3 += 1

            # Unique word indication.
            if new_word:
                if senses_count == considered_sense_count and not lemma in words_checked_b1:
                    words_checked_b3.add(lemma) # marked as checked with all info on all senses

                if sense_match(sense_id, true_sense):
                    good_a3_uq += 1
                    if senses_count == considered_sense_count:
                        good_b3_uq += 1
        elif output_prefix is not None:
            print('{},{},{},{}'.format(lemma, tag, '?', '?'), file=out3)

        if decision4 is not None:
            sense_id, prob, senses_count, considered_sense_count = decision4
            
            if output_prefix is not None:
                sense_data = sense_id.split('_')
                print('{},{},{},{}'.format(lemma, tag, sense_data[2], sense_data[0]), file=out4)

            # Increase occurence counts.
            num_a4 += 1
            if senses_count == considered_sense_count:
                num_b4 += 1

            # Correct?
            if sense_match(sense_id, true_sense):
                good_a4 += 1
                if senses_count == considered_sense_count:
                    good_b4 += 1

            # Unique word indication.
            if new_word:
                if senses_count == considered_sense_count and (not lemma in words_checked_b1
                        and not lemma in words_checked_b2 and not lemma in words_checked_b3):
                    words_checked_b4.add(lemma) # marked as checked with all info on all senses

                if sense_match(sense_id, true_sense):
                    good_a4_uq += 1
                    if senses_count == considered_sense_count:
                        good_b4_uq += 1
        elif output_prefix is not None:
            print('{},{},{},{}'.format(lemma, tag, '?', '?'), file=out4)

        if use_descriptions:
            if decision5 is not None:
                sense_id, prob, senses_count, considered_sense_count = decision5
                if output_prefix is not None:
                    sense_data = sense_id.split('_')
                    print('{},{},{},{}'.format(lemma, tag, sense_data[2], sense_data[0]), file=out5)
                
                # Increase occurence counts.
                num_a5 += 1
                if senses_count == considered_sense_count:
                    num_b5 += 1

                # Correct?
                if sense_match(sense_id, true_sense):
                    good_a5 += 1
                    good5 = True
                    if senses_count == considered_sense_count:
                        good_b5 += 1

                # Unique word indication.
                if new_word:
                    if senses_count == considered_sense_count:
                        words_checked_b5.add(lemma) # marked as checked with all info on all senses

                    if sense_match(sense_id, true_sense):
                        good_a5_uq += 1
                        if senses_count == considered_sense_count:
                            good_b5_uq += 1
            if output_prefix is not None:
                print('{},{},{},{}'.format(lemma, tag, '?', '?'), file=out5)

        # Re-run with diagnostics when diagnostics_when_23_fails condition is true.
        if diagnostics_when_23_fails and (not full_diagnostics) and good1 and not (good2 and good3):
            predict_sense(lemma, tag, [tok_info[0] for tok_info in sent], # give only lemmas
                    tid, verbose=True)
            print('The true sense was', true_sense)

if output_prefix is not None:
    out1.close()
    out2.close()
    out3.close()
    out4.close()

num_all_unique = len(words_checked_a)+len(words_checked_rest) # "rest" only captures fundamental prediction failure

print()
print('Relations subset 1 (some, all senses present): all | good | accuracy')
print('all occurences (also when no decision)', num_all, good_a1, good_a1/num_all)
print('all occurences, unique', num_all_unique, good_a1_uq, good_a1_uq/num_all_unique)
print('----')
print(num_b1, good_b1, good_b1/num_b1)
print('unique', len(words_checked_b1), good_b1_uq, good_b1_uq/len(words_checked_b1))

print()
print('Relations subset 2 (some, all senses present): all | good | accuracy')
print('all occurences', num_all, good_a2, good_a2/num_all)
print('all occurences, unique', num_all_unique, good_a2_uq, good_a2_uq/num_all_unique)
print('----')
print(num_b2, good_b2, good_b2/num_b2)
print('unique', len(words_checked_b2)+len(words_checked_b1), good_b2_uq,
        good_b2_uq/(len(words_checked_b2)+len(words_checked_b1)))

print()
print('Relations subset 3 (some, all senses present): all | good | accuracy')
print('all occurences', num_all, good_a3, good_a3/num_all)
print('all occurences, unique', num_all_unique, good_a3_uq, good_a3_uq/num_all_unique)
print('----')
print(num_b3, good_b3, good_b3/num_b3)
print('unique', len(words_checked_b3)+len(words_checked_b1), good_b3_uq,
        good_b3_uq/(len(words_checked_b3)+len(words_checked_b1)))

print()
print('Relations subset 4 (some, all senses present): all | good | accuracy')
print('all occurences', num_all, good_a4, good_a4/num_all)
print('all occurences, unique', num_all_unique, good_a4_uq, good_a4_uq/num_all_unique)
print('----')
print(num_b4, good_b4, good_b4/num_b4)
print('unique', len(words_checked_b4)+len(words_checked_b3)+len(words_checked_b2)+len(words_checked_b1),
        good_b4_uq,
        good_b4_uq/(len(words_checked_b4)+len(words_checked_b3)+len(words_checked_b2)+len(words_checked_b1)))

if use_descriptions:
    print()
    print('Descriptions (some, all senses present): all | good | accuracy')
    print('all occurences', num_all, good_a5, good_a5/num_all)
    print('all occurences, unique', num_all_unique, good_a5_uq, good_a5_uq/num_all_unique)
    print('----')
    print(num_b5, good_b5, good_b5/num_b5)
    print('unique', len(words_checked_b5), good_b5_uq, good_b5_uq/len(words_checked_b5))
