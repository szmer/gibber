import os

from operator import itemgetter
from lxml import etree

from resources_conf import nkjp_index_path, skladnica_sections_index_path, skladnica_path

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


# ## Tests

##
## Collect model decisions for all subsets of relations, counting correct decisions.
##

## numbers indicate the relation group,
## a - words where at least one sense has known neighbors,
## b - words where all senses have known neighbors,
## uq - no word repetitions
## "Good" values will be divided by N to get accuracy.
a_words_checked = []
b1_words_checked = []
b2_words_checked = []
b3_words_checked = []
b4_words_checked = []
num_a = 0
num_b = 0
# also, len(words_checked)
good1a = 0.0
good1b = 0.0
good2a = 0.0
good2b = 0.0
good3a = 0.0
good3b = 0.0
good4a = 0.0
good4b = 0.0

print('Group 1 (some, all senses present)')
print(len(a_words_checked), good1a/len(a_words_checked))
print(len(b1_words_checked), good1a/len(b1_words_checked))
print('Group 2 (some, all senses present)')
print(len(a_words_checked), good2a/len(a_words_checked))
print(len(b2_words_checked), good1a/len(b2_words_checked))
print('Group 3 (some, all senses present)')
print(len(a_words_checked), good3a/len(a_words_checked))
print(len(b3_words_checked), good1a/len(b3_words_checked))
print('Group 4 (some, all senses present)')
print(len(a_words_checked), good4a/len(a_words_checked))
print(len(b4_words_checked), good1a/len(b4_words_checked))
