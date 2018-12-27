import os, csv, re
from itertools import chain

from operator import itemgetter
from lxml import etree

from resources_conf import nkjp_index_path, mode, skladnica_sections_index_path, skladnica_path, annot_sentences_path
from gibber_wsd_model import add_word_neighborhoods, fill_sample, predict_sense
import gibber_wsd_model

# Transform lemmas of some verb forms?
gibber_wsd_model.TRANSFORM_LEMMAS = True # False

# For referring to wordnet senses, we use a convention of symbols aaa_111, where
# aaa is the lexical id, and 111 is the variant number. If one of these is empty
# (as it is in the loaded sense-annotated data), it is enough for the other
# information to match.

### Get Składnica sentences

# NOTE this also contains empty lists where the sents were actually not annotated.
sents = [] # pairs: (word lemma, lexical unit id [or None])
words = set() # all unique words that are present

if mode == 'wordnet2_annot':
    skl_ids = [] # document identifiers

    with open(skladnica_sections_index_path) as index:
        for sec_name in index:
            skl_ids.append(sec_name.strip())

    ## Get all the sentences present in Składnica.

    for skl_id in skl_ids:
        section_path = skladnica_path + skl_id
        # Each section has 1+ dirs of XML files.
        for dirname, _, __ in os.walk(section_path):
            if dirname != section_path: # ignore the dir itself
                # Get the XML files (each contains a sentence)
                for _, __, filenames in os.walk(dirname):
                    for filename in filenames:
                        tree = etree.parse(dirname+'/'+filename)
                        # Check if the file has wordnet annotations.
                        if tree.find('.//plwn_interpretation') is None:
                            continue
                        
                        sent_id = filename[:-len('.xml')]
                        
                        # Collect the list of sentence lemmas.
                        sent = []
                        for elem in tree.iterfind('node[@chosen]'):
                            if elem.attrib['chosen'] == 'true' and elem.find("terminal") is not None:
                                lexical_id = None
                                if elem.find('.//luident') is not None: # if word-sense annotation is available
                                    lexical_id = elem.find('.//luident').text+'_'

                                lemma = elem.find("terminal").find("base").text
                                if lemma is None: # happens for numbers
                                    lemma = elem.find("terminal").find("orth").text
                                lemma = lemma.lower()
                                tag = elem.find('.//f[@type]').text # should be <f type="tag">
                                sent.append((int(elem.attrib['from']), lemma, lexical_id, tag))
                                words.add(lemma)
                        sent.sort(key=itemgetter(0))
                        sent = [(token, lexical_id, tag) for num, token, lexical_id, tag in sent]
                        sents.append(sent)

if mode == 'wordnet3_annot':
    with open(annot_sentences_path, newline='') as annot_file:
        annot_reader = csv.reader(annot_file)
        sent = []
        for row in annot_reader:
            form, lemma, tag, true_sense = row[0], row[1], row[2], row[3]
            if form == '&' and lemma == '\\&\\':
                sents.append(sent)
                sent = []
            else:
                if re.match('\\d+', true_sense):
                    sent.append((lemma.lower(), '_'+true_sense, tag))
                else:
                    sent.append((lemma.lower(), None, tag))
                words.add((lemma.lower(), tag))


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
## a - words where at least one sense has known neighbors,
## b - words where all senses have known neighbors,
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

words_checked_rest = set() # for calculating the total performance
num_all = 0 # number of all occurences where we have some true sense provided

num_a1 = 0
num_a2 = 0
num_a3 = 0
num_a4 = 0
num_b1 = 0
num_b2 = 0
num_b3 = 0
num_b4 = 0
# also, len(words_checked)
good_a1 = 0.0
good_b1 = 0.0
good_a2 = 0.0
good_b2 = 0.0
good_a3 = 0.0
good_b3 = 0.0
good_a4 = 0.0
good_b4 = 0.0
good_a1_uq = 0.0
good_b1_uq = 0.0
good_a2_uq = 0.0
good_b2_uq = 0.0
good_a3_uq = 0.0
good_b3_uq = 0.0
good_a4_uq = 0.0
good_b4_uq = 0.0

##
## Collect model decisions for all subsets of relations, counting correct decisions.
##

def sense_match(full, partial):
    full = full.split('_')
    lexid, variant = full[0], full[1]
    return re.match('^({})?_({})?$'.format(lexid, variant), partial)

print('Performing the test on sense-annotated sentences.')
for sent in sents:
    for (tid, token_data) in enumerate(sent):
        lemma, true_sense, tag = token_data[0], token_data[1], token_data[2]
        if true_sense is None:
            continue

        fill_sample(lemma, sent, tid)
        try:
            decision1, decision2, decision3, decision4 = predict_sense(lemma, tag, sent, tid)
        except LookupError as err:
            print(err)
            words_checked_rest.add(lemma)
            continue

        num_all += 1
        if [None, None, None, None] == [decision1, decision2, decision3, decision4]:
            words_checked_rest.add(lemma)
        
        new_word = False

        if decision1 is not None:
            sense_id, prob, senses_count, considered_sense_count = decision1
            
            # Increase occurence counts.
            num_a1 += 1
            if senses_count == considered_sense_count:
                num_b1 += 1

            # Correct?
            if sense_match(sense_id, true_sense):
                good_a1 += 1
                if senses_count == considered_sense_count:
                    good_b1 += 1

            # Unique word indication.
            if not lemma in words_checked_a:
                words_checked_a.add(lemma)
                new_word = True

                if senses_count == considered_sense_count:
                    words_checked_b1.add(lemma) # marked as checked with all info on all senses

                if sense_match(sense_id, true_sense):
                    good_a1_uq += 1
                    if senses_count == considered_sense_count:
                        good_b1_uq += 1

        if decision2 is not None or decision1 is not None:
            if decision2 is None:
                sense_id, prob, senses_count, considered_sense_count = decision1
            else:
                sense_id, prob, senses_count, considered_sense_count = decision2
            
            # Increase occurence counts.
            num_a2 += 1
            if senses_count == considered_sense_count:
                num_b2 += 1

            # Correct?
            if sense_match(sense_id, true_sense):
                good_a2 += 1
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

        if decision3 is not None or decision1 is not None:
            if decision3 is None:
                sense_id, prob, senses_count, considered_sense_count = decision1
            else:
                sense_id, prob, senses_count, considered_sense_count = decision3
            
            # Increase occurence counts.
            num_a3 += 1
            if senses_count == considered_sense_count:
                num_b3 += 1

            # Correct?
            if sense_match(sense_id, true_sense):
                good_a3 += 1
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

        if decision4 is not None:
            sense_id, prob, senses_count, considered_sense_count = decision4
            
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

num_all_unique = len(words_checked_a)+len(words_checked_rest)

print()
print('Relations subset 1 (some, all senses present): all | good | accuracy')
print(num_a1, good_a1, good_a1/num_a1)
print('all occurences', num_all, good_a1, good_a1/num_all)
print('unique', len(words_checked_a), good_a1_uq, good_a1_uq/len(words_checked_a))
print('all occurences, unique', num_all_unique, good_a1_uq, good_a1_uq/num_all_unique)
print('----')
print(num_b1, good_b1, good_b1/num_b1)
print('all occurences', num_all, good_b1, good_b1/num_all)
print('unique', len(words_checked_b1), good_b1_uq, good_b1_uq/len(words_checked_b1))
print('all occurences, unique', num_all_unique, good_b1_uq, good_b1_uq/num_all_unique)

print()
print('Relations subset 2 (some, all senses present): all | good | accuracy')
print(num_a2, good_a2, good_a2/num_a2)
print('all occurences', num_all, good_a2, good_a2/num_all)
print('unique', len(words_checked_a), good_a2_uq, good_a2_uq/len(words_checked_a))
print('all occurences, unique', num_all_unique, good_a2_uq, good_a2_uq/num_all_unique)
print('----')
print(num_b2, good_b2, good_b2/num_b2)
print('all occurences', num_all, good_b2, good_b2/num_all)
print('unique', len(words_checked_b2)+len(words_checked_b1), good_b2_uq,
        good_b2_uq/(len(words_checked_b2)+len(words_checked_b1)))
print('all occurences, unique', num_all_unique, good_b2_uq, good_b2_uq/num_all_unique)

print()
print('Relations subset 3 (some, all senses present): all | good | accuracy')
print(num_a3, good_a3, good_a3/num_a3)
print('all occurences', num_all, good_a3, good_a3/num_all)
print('unique', len(words_checked_a), good_a3_uq, good_a3_uq/len(words_checked_a))
print('all occurences, unique', num_all_unique, good_a3_uq, good_a3_uq/num_all_unique)
print('----')
print(num_b3, good_b3, good_b3/num_b3)
print('all occurences', num_all, good_b3, good_b3/num_all)
print('unique', len(words_checked_b3)+len(words_checked_b1), good_b3_uq,
        good_b3_uq/(len(words_checked_b3)+len(words_checked_b1)))
print('all occurences, unique', num_all_unique, good_b3_uq, good_b3_uq/num_all_unique)

print()
print('Relations subset 4 (some, all senses present): all | good | accuracy')
print(num_a4, good_a4, good_a4/num_a4)
print('all occurences', num_all, good_a4, good_a4/num_all)
print('unique', len(words_checked_a), good_a4_uq, good_a4_uq/len(words_checked_a))
print('all occurences, unique', num_all_unique, good_a4_uq, good_a4_uq/num_all_unique)
print('----')
print(num_b4, good_b4, good_b4/num_b4)
print('all occurences', num_all, good_b4, good_b4/num_all)
print('unique', len(words_checked_b4)+len(words_checked_b3)+len(words_checked_b2)+len(words_checked_b1),
        good_b4_uq,
        good_b4_uq/(len(words_checked_b4)+len(words_checked_b3)+len(words_checked_b2)+len(words_checked_b1)))
print('all occurences, unique', num_all_unique, good_b4_uq, good_b4_uq/num_all_unique)
