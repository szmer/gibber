from collections import defaultdict


index_file = "/home/szymon/lingwy/nkjp/nkjp_index.txt"
nkjp_path = "/home/szymon/lingwy/nkjp/pełny/"
vecs_path = "/home/szymon/lingwy/nkjp/wektory/nkjp+wiki-lemmas-all-100-skipg-ns.txt/"
vecs_dim = 100
skladnica_sections_index = "/home/szymon/lingwy/nkjp/skladnica_znacz/sections.txt"
skladnica_path = "/home/szymon/lingwy/nkjp/skladnica_znacz/"
pl_wordnet = "/home/szymon/lingwy/wielozn/wzięte/plwordnet_2_0_4/plwordnet_2_0_4_release/plwordnet_2_0.xml"

window_size = 4 # how many words to condider on both sides of the target
batch_size = window_size * 2 # for training of gibberish discriminator in Keras
learning_rate = 0.3
reg_rate = 0.005
corp_runs = 2 # how many times to feed the whole corpus during training


# ## Get Składnica & NKJP sentences


## We want to cross-identify Składnica and NKJP documents.

skl_ids = [] # document identificators
nkjp_orig_ids = []
nkjp_squeezed_ids = [] # Składnica uses "squeezed" version of ids, w/o hyphens (-)

# This will be used later. NKJP id -> [ its sent_ids ]
skl_nkjp_sent_ids = {}
# auxiliary, skl_id -> NKJP_id for documents
nkjp_skl_ids = {}

with open(skladnica_sections_index) as index:
    for sec_name in index:
        skl_ids.append(sec_name.strip())
with open(index_file) as index:
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



from lxml import etree
import os



## Get all the sentences present in Składnica.
from operator import itemgetter

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



skl_sents[:6]



## Read the NKJP fragments, as lists of lemmas
unique_words = set()
words_count = 0
train_sents = []
rejected_sent_ids = []

with open(index_file) as index:
    for fragm_id in index:
        fragm_id = fragm_id.strip()
        # tag is namespaced, .// for finding anywhere in the tree
        tokens_filepath = nkjp_path+fragm_id+'/ann_morphosyntax.xml'
        if not os.path.isfile(tokens_filepath):
            print('Note: cannot access {}'.format(fragm_id.strip()))
            continue
        tree = etree.parse(tokens_filepath)
        for sent_subtree in tree.iterfind('.//{http://www.tei-c.org/ns/1.0}s[@{http://www.w3.org/XML/1998/namespace}id]'):
            
            # Skip sentences analyzed in Składnica.
            sent_id = sent_subtree.get('{http://www.w3.org/XML/1998/namespace}id').replace(' ', '_')
            if fragm_id in skl_nkjp_sent_ids and sent_id in skl_nkjp_sent_ids[fragm_id]:
                rejected_sent_ids.append(fragm_id+':'+sent_id)
                continue
                
            sent_lemmas = []
            for seg in sent_subtree.iterfind('.//{http://www.tei-c.org/ns/1.0}seg'):
                form = None
                lemma = None 
                for elem in seg.iterfind('.//{http://www.tei-c.org/ns/1.0}f[@name]'):
                    if elem.attrib['name'] == 'base':
                        lemma = elem[0].text # first child <string>
                    if elem.attrib['name'] == 'orth':
                        form = elem[0].text # first child <string>
                if lemma is not None:
                    sent_lemmas.append(lemma.lower())
                else: # can happen when the form is number and there's no lemma
                    sent_lemmas.append(form.lower())
            train_sents += [[]]
            for word in sent_lemmas:
                train_sents[-1].append(word)
                words_count += 1
                unique_words.add(word)
print('{} individual sents rejected, for {} known from test corpus'.format(len(rejected_sent_ids), len(skl_sents)))



words_count


# ## Get word vectors


import numpy as np



first_line = True
word_n = 0
word_to_idx = {} # ie. indices of vectors
idx_to_word = {}

# we'll read those from the data file
vecs_count = 0
vecs_dim = 100

# Get the vector word labels (we'll get vectors themselves in a momment).
with open(vecs_path+"data") as vecs_file:
    for line in vecs_file:
        if first_line:
            # Read metadata.
            vecs_count = int(line.split(' ')[0])
            vecs_dim = int(line.split(' ')[1])
            first_line = False
            continue
        # Read lemma base forms.
        word = line.split(' ')[0].lower()
        word_to_idx[word] = word_n
        idx_to_word[word_n] = word
        word_n += 1



word_vecs = np.loadtxt(vecs_path+"data", encoding="utf-8",
                       dtype=np.float32, # tensorflow's requirement
                       skiprows=1, usecols=tuple(range(1, vecs_dim+1)))



# Add the dummy boundary/unknown marker.
word_vecs = np.vstack([word_vecs, np.zeros((1,vecs_dim), dtype=np.float32)])
vecs_count += 1



# Get the word's vector, or the dummy marker.
def word_id(word):
    return word_to_idx[word] if word in word_to_idx else vecs_count-1



vecs_count, vecs_dim


# ## Keras - training the "language/gibberish discriminator"


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Embedding
from keras.optimizers import SGD



##
## Training corpus preparation.
##
## We want a training set of multiword contexts (train) and target words (labels);
## the model will learn to distinguish genuine from fake (negative) context-target pairs

from random import randint
from math import floor

# We need a special token for cases when the target word is near the start or end of sentence.
bound_token_id = vecs_count - 1 # the zero additional vector

sample_n = 0

train = np.zeros(((words_count + words_count) * corp_runs, # positive & negative samples
                  window_size * 2 + 1), dtype='int')
labels = np.ones(((words_count + words_count) * corp_runs,),
                dtype='int')

for run_n in range(corp_runs):
    sent_n = 0
    word_n = 0
        
    while sent_n < len(train_sents) and sample_n < train.shape[0]:
        # The positive sample.
        train[sample_n, window_size] = word_id(train_sents[sent_n][word_n]) # the target word
        for j in range(window_size): # its neighbors in the window_size
            train[sample_n, j] = (word_id(train_sents[sent_n][word_n-j-1]) if word_n-j-1 >= 0
                                  else bound_token_id)
            train[sample_n, window_size+j+1] = (word_id(train_sents[sent_n][word_n+j+1])
                                                if word_n+j+1 < len(train_sents[sent_n])
                                                else bound_token_id)
        # (the "one" positive label is the default)
        # labels[sample_n] = 1.0
        
        # The negative sample.
        sample_n += 1
        train[sample_n,] = np.random.permutation(train[sample_n-1,])
        # (replace two random words)
        train[sample_n, randint(0, window_size*2)] = randint(0, vecs_count-1)
        train[sample_n, randint(0, window_size*2)] = randint(0, vecs_count-1)
        labels[sample_n] = 0.0
                
        sample_n += 1
        word_n += 1
        # If words are exhausted, scan the sents for one that has some words.
        try:
            while word_n == len(train_sents[sent_n]):
                word_n = 0
                sent_n += 1
        # At the end of the corpus (== exhausted sents), break the loop.
        except IndexError:
            break



model = Sequential()                                                                                               
model.add(Embedding(vecs_count,
                    vecs_dim,
                    weights=[word_vecs],
                    input_length=window_size * 2 + 1,
                    trainable=False))                                                                              
model.add(LSTM(96))
model.add(Dense(1))
model.add(Activation('sigmoid'))

opt = SGD(lr=learning_rate, decay=reg_rate)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])



model.fit(train, labels)



model



# cleanup to save some memory
train, labels = None, None


# ## Loading Polish Wordnet 2 information


# Relations we're interested in:
# only hyper/hyponymy, similarity, synonymy
rels1 = { '60', '11', '30' }
# should be rels1 + antonymy and conversion
rels2 = { '12', '101', '102', '104', '13' }
# should be rels1 + various meronymies and meronymy-like relations
rels3 = {'14', '15' # mero/holonymy
         '20', '21', '22', '23', '24', # meronymy types
         '64', '65', '80', # taxonomic type, causation
         '106', # type
         # verb meronymies
         '112', '113', '114', '115', '119', '120', '121', '122', '128', '129',
         '130', '131', '132', '134', '136', '137', '138', '140',
         # adjective relations
         '147', '150', '151', '152' }
# all relations
# rels4 is rels2 plus rels3

## Dynamic storage of word relations info
wordnet_xml = etree.parse(pl_wordnet)

# NOTE. Theoretically, we have lex_unit ids and synsets given by annotation, but we want
# to take into account all possibilities given by raw word forms. Our model will try to
# **select** the correct lex_unit ids for given form in given context.
# NOTE 2. We store the bulk of relations in x1 dicts, x2 and x3 store only ones specific to them

# NOTE 3. "lexical units" and "wordids" in fact relate to distinct senses, as per the Wordnet.

skl_word_wordids = defaultdict(list) # word -> ids as list

skl_wordid_neighbors1 = defaultdict(list) # lemmas
skl_wordid_neighbors2 = defaultdict(list)
skl_wordid_neighbors3 = defaultdict(list)

# go from "a => [v1, v2 ... ]" to "v1 => [ a ... ], v2 => [ a ... ] ..."
def reverse_list_dict(dic):
    new_dic = defaultdict(list)
    for key, val in dic.items():
        for v in val:
            new_dic[v].append(key)
    return new_dic


## Dealing with some lemmatization quirks of our corpus.
def rm_sie(base):
    if base[-len(' się'):] == ' się':
        return base[:-len(' się')]
    else:
        return base
# Some lemmas mapped into lexical unit of "osoba"
persons = ['ja', 'ty', 'on', 'ona', 'ono', 'my', 'wy', 'oni', 'one', 'siebie', 'kto', 'któż', 'wszyscy']

def add_word_neighborhoods(words):
    """Add neighbor forms for given words to skl_wordid_neighbors* variables."""

    words = set(['osoba' if (w in persons) else w for w in words])

    new_words = set([w for w in words if not w in word_wordids])

    # Prepare the temporary indices.
    skl_wordid_synsets = defaultdict(list) # to harness synset relations
    skl_neighbor_syns1 = defaultdict(list) # just for collecting them, synset ids
    skl_neighbor_syns2 = defaultdict(list)
    skl_neighbor_syns3 = defaultdict(list)
    skl_wordid_neighbor_ids1 = defaultdict(list) # just for collecting them
    skl_wordid_neighbor_ids2 = defaultdict(list)
    skl_wordid_neighbor_ids3 = defaultdict(list)

    if len(new_words) > 0:
        ## Collect word ids and synsets for new words.
        # Wordnet lexical ID's of lemmas present in Składnica.
        for lex_unit in wordnet_xml.iterfind('lexical-unit'):
            form = lex_unit.get('name').lower()
            if form in words or rm_sie(form) in new_words:
                word_wordids[form].append(lex_unit.get('id'))

        # The r.* dicts are obtained by reversal of their parent dictionary when needed by the neighborhood function.
        # (reverse dict)
        rskl_word_wordids = reverse_list_dict(skl_word_wordids) # ids -> words

        # !!! From now on, make sure that this dict only contains new_words.
        rskl_word_wordids = dict([(id, word) for (id, word) in rskl_word_wordids if word in new_words])

        # Get the synsets for new words.
        for synset in wordnet_xml.iterfind('synset'):
            for unit in synset.iterfind('unit-id'):
                unit_word = unit.text.lower()
                if unit_word in rskl_word_wordids:
                    skl_wordid_synsets[unit.text.lower()].append(synset.get('id')) # assign synset id to the wordid

        ## Collect lexical relations for new words.
        for lexrel in wordnet_xml.iterfind('lexicalrelations'):
            if lexrel.get('parent') in rskl_word_wordids:
                if lexrel.get('relation') in rels1:
                    skl_wordid_neighbor_ids1[lexrel.get('parent')].append(lexrel.get('child'))
                if lexrel.get('relation') in rels2:
                    skl_wordid_neighbor_ids2[lexrel.get('parent')].append(lexrel.get('child'))
                if lexrel.get('relation') in rels3:
                    skl_wordid_neighbor_ids3[lexrel.get('parent')].append(lexrel.get('child'))
            if lexrel.get('child') in rskl_word_wordids:
                if lexrel.get('relation') in rels1:
                    skl_wordid_neighbor_ids1[lexrel.get('child')].append(lexrel.get('parent'))
                if lexrel.get('relation') in rels2:
                    skl_wordid_neighbor_ids2[lexrel.get('child')].append(lexrel.get('parent'))
                if lexrel.get('relation') in rels3:
                    skl_wordid_neighbor_ids3[lexrel.get('child')].append(lexrel.get('parent'))

        # (reverse dict)
        rskl_wordid_synsets = reverse_list_dict(skl_wordid_synsets)

        # Get synset relations.  for synrel in wordnet_xml.iterfind('synsetrelations'):
            if synrel.get('parent') in rskl_wordid_synsets:
                if synrel.get('relation') in rels1:
                    skl_neighbor_syns1[synrel.get('child')].append(synrel.get('parent'))
                if synrel.get('relation') in rels2:
                    skl_neighbor_syns2[synrel.get('child')].append(synrel.get('parent'))
                if synrel.get('relation') in rels3:
                    skl_neighbor_syns3[synrel.get('child')].append(synrel.get('parent'))
            if synrel.get('child') in rskl_wordid_synsets:
                if synrel.get('relation') in rels1:
                    skl_neighbor_syns1[synrel.get('parent')].append(synrel.get('child'))
                if synrel.get('relation') in rels2:
                    skl_neighbor_syns2[synrel.get('parent')].append(synrel.get('child'))
                if synrel.get('relation') in rels3:
                    skl_neighbor_syns3[synrel.get('parent')].append(synrel.get('child'))

        # Get additional neighbor wordids from neighbor synsets.
        for synset in wordnet_xml.iterfind('synset'):
            if synset.get('id') in skl_neighbor_syns1:
                # for words being in that synset
                for unit in synset.iterfind('unit-id'):
                    # for synsets for which we want to collect this one
                    for target_syns in skl_neighbor_syns1[synset.get('id')]:
                        # for wordids that are in that other synset - add this wordid from this synset
                        for receiver in rskl_wordid_synsets[target_syns]:
                            skl_wordid_neighbor_ids1[receiver].append(unit.text.lower())
            if synset.get('id') in skl_neighbor_syns2:
                for unit in synset.iterfind('unit-id'):
                    for target_syns in skl_neighbor_syns2[synset.get('id')]:
                        for receiver in rskl_wordid_synsets[target_syns]:
                            skl_wordid_neighbor_ids2[receiver].append(unit.text.lower())
            if synset.get('id') in skl_neighbor_syns1:
                for unit in synset.iterfind('unit-id'):
                    for target_syns in skl_neighbor_syns3[synset.get('id')]:
                        for receiver in rskl_wordid_synsets[target_syns]:
                            skl_wordid_neighbor_ids3[receiver].append(unit.text.lower())

        # (get reverses of these)
        rskl_wordid_neighbor_ids1 = reverse_list_dict(skl_wordid_neighbor_ids1)
        skl_neighbor_id_wordids2 = reverse_list_dict(skl_wordid_neighbor_ids2)
        skl_neighbor_id_wordids3 = reverse_list_dict(skl_wordid_neighbor_ids3)

        # Finally, harness neighbors by their known wordids.
        for lex_unit in wordnet_xml.iterfind('lexical-unit'):
            if lex_unit.get('id') in skl_neighbor_id_wordids1:
                for receiver in skl_neighbor_id_wordids1[lex_unit.get('id')]: # words "interested" in this neighbor
                    skl_wordid_neighbors1[receiver].append(lex_unit.get('name').lower())
            if lex_unit.get('id') in skl_neighbor_id_wordids2:
                for receiver in skl_neighbor_id_wordids2[lex_unit.get('id')]:
                    skl_wordid_neighbors2[receiver].append(lex_unit.get('name').lower())
            if lex_unit.get('id') in skl_neighbor_id_wordids3:
                for receiver in skl_neighbor_id_wordids3[lex_unit.get('id')]:
                    skl_wordid_neighbors3[receiver].append(lex_unit.get('name').lower())


# free some memory.
#skl_neighbor_syns_wordids1 = None
#skl_neighbor_syns_wordids2 = None
#skl_neighbor_syns_wordids3 = None
#rskl_wordid_synsets = None



# ## Tests


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



sample = np.zeros((1, window_size * 2 + 1), dtype='int')
def fill_sample(lemma, sent, tid):
    sample[0, window_size] = word_id(lemma) # the target word
    for j in range(window_size): # its neighbors in the window_size
        sample[0, j] = (word_id(sent[tid-j-1])
                        if tid-j-1 >= 0
                        else bound_token_id)
        sample[0, window_size+j+1] = (word_id(sent[tid+j+1])
                                        if tid+j+1 < len(sent)
                                        else bound_token_id)



from random import choice



##
## Collect model decisions for all subsets of relations, counting correct decisions.
##

for sent in skl_sents:
    for (tid, token) in enumerate(sent): # these are still actually lemmas
        lemma, true_sense = token[0], token[1]
        if true_sense is None:
            continue
            
        fill_sample(lemma, sent, tid)
        reference_prob = model.predict(sample)[0][0]
        
        # For the purpose of finding neighbors, we need to do these adjustments.
        if lemma in persons:
            lemma = 'osoba'
            
        sense_wordids = skl_word_wordids[lemma]
        if len(sense_wordids) == 0:
            sense_wordids = skl_word_wordids[lemma + ' się']
            if len(sense_wordids) == 0:
                print(lemma+ ' (się)', true_sense, '- no senses found, bad lemma?')
                continue
        
        # Neighbors of each sense.
        sense_ngbs1 = [skl_wordid_neighbors1[swi] for swi in sense_wordids]
        # Number of neighbors for each sense - important when calculating average for later relation subsets.
        sense_ngbcounts1 = [len(skl_wordid_neighbors1[swi]) for swi in sense_wordids]
        sense_probs1 = [reference_prob for x in sense_wordids]
        senses_considered1 = 0
        sense_ngbs2 = [skl_wordid_neighbors2[swi] for swi in sense_wordids]
        sense_ngbcounts2 = [len(skl_wordid_neighbors2[swi]) for swi in sense_wordids]
        sense_probs2 = [reference_prob for x in sense_wordids]
        senses_considered2 = 0
        sense_ngbs3 = [skl_wordid_neighbors3[swi] for swi in sense_wordids]
        sense_ngbcounts3 = [len(skl_wordid_neighbors3[swi]) for swi in sense_wordids]
        sense_probs3 = [reference_prob for x in sense_wordids]
        senses_considered3 = 0
        sense_probs4 = [reference_prob for x in sense_wordids]
        senses_considered4 = 0
        
        decision1, decision2, decision3, decision4 = '', '', '', ''
        decision1_made = False
        
        # 1st subset:
        if sum([len(x) for x in sense_ngbs1]) > 0: # we need a neighbor for at least one sense
            
            decision1_made = True
            
            for (sid, ngbs) in enumerate(sense_ngbs1):
                if len(ngbs) == 0:
                    continue
                senses_considered1 += 1
                for ngb_word in ngbs:
                    fill_sample(ngb_word, sent, tid)
                    sense_probs1[sid] += model.predict(sample)[0][0]
                    
                # We have average prob of any neighbor of this sense
                sense_probs1[sid] /= sense_ngbcounts1[sid]
                
            winners1 = [sid for (sid, p) in enumerate(sense_probs1) if p == max(sense_probs1)]
            #print(winners, sense_wordids)
            decision1 = sense_wordids[choice(winners1)]
            
            # If good, update good counts
            if decision1 == true_sense:
                good1a += 1.0
                if senses_considered1 == len(sense_wordids):
                    good1b += 1.0
            
            # Update word lists
            a_words_checked.append(lemma)
            if senses_considered1 == len(sense_wordids):
                b1_words_checked.append(lemma)
        
        # 2nd subset (1+2)
        if sum([len(x) for x in sense_ngbs2]) > 0: # there's a neighbor for at least one sense
            for (sid, ngbs) in enumerate(sense_ngbs2):
                if len(ngbs) == 0:
                    if sense_ngbcounts1[sid] > 1: # just carry the "1" average
                        sense_probs2[sid] = sense_probs1[sid]
                        senses_considered2 += 1
                    continue
                senses_considered2 += 1
                for ngb_word in ngbs:
                    fill_sample(ngb_word, sent, tid)
                    sense_probs2[sid] += model.predict(sample)[0][0]
                # We have average prob of any neighbor of this sense + carry the input from "1"
                # (note that it can be ignored due to zero neighbors)
                sense_probs2[sid] = (((sense_probs1[sid] * sense_ngbcounts1[sid]) + sense_probs2[sid])
                                    / (sense_ngbcounts1[sid] + sense_ngbcounts2[sid]))
                
            winners2 = [sid for (sid, p) in enumerate(sense_probs2) if p == max(sense_probs2)]
            decision2 = sense_wordids[choice(winners2)]
            
            if decision2 == true_sense:
                good2a += 1.0
                if senses_considered2 == len(sense_wordids):
                    good2b += 1.0
            if senses_considered2 == len(sense_wordids):
                b2_words_checked.append(lemma)
        # Carry the "1" decision for cases without specific "2" info.
        elif decision1_made:
            if decision1 == true_sense:
                good2a += 1.0
                if senses_considered1 == len(sense_wordids):
                    good2b += 1.0
            if senses_considered1 == len(sense_wordids):
                b2_words_checked.append(lemma)
               
        # 3rd subset (1+3)
        if sum([len(x) for x in sense_ngbs3]) > 0: # there's a neighbor for at least one sense
            for (sid, ngbs) in enumerate(sense_ngbs3):
                if len(ngbs) == 0:
                    if sense_ngbcounts1[sid] > 1: # just carry the "1" average
                        sense_probs3[sid] = sense_probs1[sid]
                        senses_considered3 += 1
                    continue
                senses_considered3 += 1
                for ngb_word in ngbs:
                    fill_sample(ngb_word, sent, tid)
                    sense_probs3[sid] += model.predict(sample)[0][0]
                # We have average prob of any neighbor of this sense + carry the input from "1"
                # (note that it can be ignored due to zero neighbors)
                sense_probs3[sid] = (((sense_probs1[sid] * sense_ngbcounts1[sid]) + sense_probs3[sid])
                                    / (sense_ngbcounts1[sid] + sense_ngbcounts3[sid]))
                
            winners3 = [sid for (sid, p) in enumerate(sense_probs3) if p == max(sense_probs3)]
            decision3 = sense_wordids[choice(winners3)]
            
            if decision3 == true_sense:
                good3a += 1.0
                if senses_considered3 == len(sense_wordids):
                    good3b += 1.0
            if senses_considered3 == len(sense_wordids):
                b3_words_checked.append(lemma)
        # Carry the "1" decision for cases without specific "3" info.
        elif decision1_made:
            if decision1 == true_sense:
                good3a += 1.0
                if senses_considered1 == len(sense_wordids):
                    good3b += 1.0
            if senses_considered1 == len(sense_wordids):
                b3_words_checked.append(lemma)
                
        # 4th subset (1+2+3)
        for sid, _ in enumerate(sense_wordids):
            if sense_ngbcounts1[sid] == 0 and sense_ngbcounts2[sid] == 0 and sense_ngbcounts3[sid] == 0:
                continue
            senses_considered4 += 1
            sense_probs4[sid] = ((sense_probs1[sid] * sense_ngbcounts1[sid]
                                 + sense_probs2[sid] * sense_ngbcounts2[sid]
                                 + sense_probs3[sid] * sense_ngbcounts3[sid])
                                / (sense_ngbcounts1[sid] + sense_ngbcounts2[sid] + sense_ngbcounts3[sid]))
        winners4 = [sid for (sid, p) in enumerate(sense_probs4) if p == max(sense_probs4)]
        decision4 = sense_wordids[choice(winners4)]
            
        if decision4 == true_sense:
            good4a += 1.0
            if senses_considered4 == len(sense_wordids):
                good4b += 1.0
        if senses_considered4 == len(sense_wordids):
            b4_words_checked.append(lemma)



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

