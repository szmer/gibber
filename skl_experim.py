import os
from itertools import chain

from operator import itemgetter
from lxml import etree

from resources_conf import nkjp_index_path, skladnica_sections_index_path, skladnica_path
from gibber_wsd_model import predict_sense

# ## Get Składnica sentences

## We want to cross-identify Składnica and NKJP documents.

skl_ids = [] # document identificators
nkjp_orig_ids = []
nkjp_squeezed_ids = [] # Składnica uses "squeezed" version of ids, w/o hyphens (-)

# This will be used later. NKJP id -> [ its sent_ids ]
skl_nkjp_sent_ids = {}
# auxiliary, skl_id -> NKJP_id for documents
nkjp_skl_ids = {}

with open(skladnica_sections_index_path) as index:
    for sec_name in index:
        skl_ids.append(sec_name.strip())
with open(nkjp_index_path) as index:
    for fragm_id in index:
        fragm_id = fragm_id.strip()
        nkjp_orig_ids.append(fragm_id)
        nkjp_squeezed_ids.append(fragm_id.replace('-', ''))
        
bonus_skladnica_mappings = {
                            'NKJP_1M_001000-DziennikPolski1980': 'DP1980',
                            'NKJP_1M_010200-ZycieWarszawyPLUSZycie': 'ZycieWarszawy_Zycie',
                            'NKJP_1M_012750-TrybunaPLUSTrybunaLudu': 'TrybunaLudu_Trybuna',
                            'NKJP_1M_1998': 'DP1998',
                            'NKJP_1M_1999': 'DP1999',
                            'NKJP_1M_2000': 'DP2000',
                            'NKJP_1M_2001': 'DP2001',
                            'NKJP_1M_2002': 'DP2002',
                            'NKJP_1M_2003': 'DP2003',
                            'NKJP_1M_%2E': '712-3-900001'
                           }

## This should print Składnica ids for which we don't know their NKJP equivalents.
## (hopefully, none)
for id in skl_ids:
    if id[len('NKJP_1M_'):] in nkjp_squeezed_ids:
        sqzd_nkjp_id = id[len('NKJP_1M_'):]
        orig_nkjp_id = nkjp_orig_ids[nkjp_squeezed_ids.index(sqzd_nkjp_id)]
        skl_nkjp_sent_ids[orig_nkjp_id] = []
        nkjp_skl_ids[id] = orig_nkjp_id
    elif id[len('NKJP_1M_4scal-'):] in nkjp_squeezed_ids:
        sqzd_nkjp_id = id[len('NKJP_1M_4scal-'):]
        orig_nkjp_id = nkjp_orig_ids[nkjp_squeezed_ids.index(sqzd_nkjp_id)]
        skl_nkjp_sent_ids[orig_nkjp_id] = []
        nkjp_skl_ids[id] = orig_nkjp_id
    elif id[len('NKJP_1M_######-'):] in nkjp_squeezed_ids:
        sqzd_nkjp_id = id[len('NKJP_1M_######-'):]
        orig_nkjp_id = nkjp_orig_ids[nkjp_squeezed_ids.index(sqzd_nkjp_id)]
        skl_nkjp_sent_ids[orig_nkjp_id] = []
        nkjp_skl_ids[id] = orig_nkjp_id
    elif id in bonus_skladnica_mappings:
        orig_nkjp_id = bonus_skladnica_mappings[id]
        skl_nkjp_sent_ids[orig_nkjp_id] = []
        nkjp_skl_ids[id] = orig_nkjp_id
    else:
        print(id)

## Get all the sentences present in Składnica.

# NOTE this also contains empty lists where the sents were actually not annotated.
skl_sents = [] # pairs: (word lemma, lexical unit id [or None])
skl_words = set() # all unique words that are present

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
                    skl_nkjp_sent_ids[nkjp_skl_ids[skl_id]].append(sent_id) # add this sentence as annonated
                    
                    # Collect the list of sentence lemmas.
                    sent = []
                    for elem in tree.iterfind('node[@chosen]'):
                        if elem.attrib['chosen'] == 'true' and elem.find("terminal") is not None:
                            lexical_id = None
                            if elem.find('.//luident') is not None: # if word-sense annotation is available
                                lexical_id = elem.find('.//luident').text

                            lemma = elem.find("terminal").find("base").text
                            if lemma is None: # happens for numbers
                                lemma = elem.find("terminal").find("orth").text
                            lemma = lemma.lower()
                            sent.append((int(elem.attrib['from']), lemma, lexical_id))
                            skl_words.add(lemma)
                    sent.sort(key=itemgetter(0))
                    sent = [(token, lexical_id) for num, token, lexical_id in sent]
                    skl_sents.append(sent)

#
# Load all the necessary wordnet data.
#
add_word_neighborhoods(set(chain.from_iterable(skl_sents)))

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

for sent in skl_sents:
    for (tid, token_data) in enumerate(sent):
        lemma, true_sense = token_data[0], token_data[1]
        if true_sense is None:
            continue

        fill_sample(lemma, sent, tid)
        decision1, decision2, decision3, decision4 = predict_sense(lemma, sent, tid)

        num_all += 1
        if [None, None, None, None] == [decision1, decision2, decision3, decision4]:
            words_checked_rest.add(lemma)
        
        new_word = False

        if decision1 is not None:
            prob, sense_id, senses_count, considered_sense_count = decision1
            
            # Increase occurence counts.
            num_a1 += 1
            if senses_count == considered_sense_count:
                num_b1 += 1

            # Correct?
            if sense_id == true_sense:
                good_a1 += 1
                if senses_count == considered_sense_count:
                    good_b1 += 1

            # Unique word indication.
            if not lemma in words_checked_a:
                word_checked_a.add(lemma)
                new_word = True

                if senses_count == considered_sense_count:
                    words_checked_b1.add(lemma) # marked as checked with all info on all senses

                if sense_id == true_sense:
                    good_a1_uq += 1
                    if senses_count == considered_sense_count:
                        good_b1_uq += 1

        if decision2 is not None:
            prob, sense_id, senses_count, considered_sense_count = decision2
            
            # Increase occurence counts.
            num_a2 += 1
            if senses_count == considered_sense_count:
                num_b2 += 1

            # Correct?
            if sense_id == true_sense:
                good_a2 += 1
                if senses_count == considered_sense_count:
                    good_b2 += 1

            # Unique word indication.
            if new_word:
                if senses_count == considered_sense_count and not lemma in words_checked_b1:
                    words_checked_b2.add(lemma) # marked as checked with all info on all senses

                if sense_id == true_sense:
                    good_a2_uq += 1
                    if senses_count == considered_sense_count:
                        good_b2_uq += 1

        if decision3 is not None:
            prob, sense_id, senses_count, considered_sense_count = decision3
            
            # Increase occurence counts.
            num_a3 += 1
            if senses_count == considered_sense_count:
                num_b3 += 1

            # Correct?
            if sense_id == true_sense:
                good_a3 += 1
                if senses_count == considered_sense_count:
                    good_b3 += 1

            # Unique word indication.
            if new_word:
                if senses_count == considered_sense_count and not lemma in words_checked_b1:
                    words_checked_b3.add(lemma) # marked as checked with all info on all senses

                if sense_id == true_sense:
                    good_a3_uq += 1
                    if senses_count == considered_sense_count:
                        good_b3_uq += 1

        if decision4 is not None:
            prob, sense_id, senses_count, considered_sense_count = decision4
            
            # Increase occurence counts.
            num_a4 += 1
            if senses_count == considered_sense_count:
                num_b4 += 1

            # Correct?
            if sense_id == true_sense:
                good_a4 += 1
                if senses_count == considered_sense_count:
                    good_b4 += 1

            # Unique word indication.
            if new_word:
                if senses_count == considered_sense_count and (not lemma in words_checked_b1
                        and not lemma in words_checked_b2 and not lemma in words_checked_b3):
                    words_checked_b4.add(lemma) # marked as checked with all info on all senses

                if sense_id == true_sense:
                    good_a4_uq += 1
                    if senses_count == considered_sense_count:
                        good_b4_uq += 1

num_all_unique = len(words_checked_b1)+len(words_checked_b2)+len(words_checked_b3)+len(words_checked_b4)+len(words_checked_rest)
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

print('Relations subset 4 (some, all senses present): all | good | accuracy')
print(num_a4, good_a4, good_a4/num_a4)
print('all occurences', num_all, good_a4, good_a4/num_all)
print('unique', len(words_checked_a), good_a4_uq, good_a4_uq/len(words_checked_a))
print('all occurences, unique', num_all_unique, good_a4_uq, good_a4_uq/num_all_unique)
print('----')
print(num_b4, good_b4, good_b4/num_b4)
print('all occurences', num_all, good_b4, good_b4/num_all)
print('unique', len(words_checked_b4)+len(words_checked_b3)+len(words_checked_b2)+len(words_checked_b1)
        good_b4_uq,
        good_b4_uq/(len(words_checked_b4)+len(words_checked_b3)+len(words_checked_b2)+len(words_checked_b1)))
print('all occurences, unique', num_all_unique, good_b4_uq, good_b4_uq/num_all_unique)
