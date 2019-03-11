import json, os, re, logging
from random import randint, shuffle, choice
from collections import defaultdict
from itertools import chain

from lxml import etree
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import torch.utils.data

from wsd_config import nkjp_format, nkjp_index_path, nkjp_path, pl_wordnet_path, model_path, window_size, corp_runs, learning_rate, reg_rate, POS_extended_model, lstm_hidden_size, lstm_layers_count, lstm_is_bidirectional, freeze_embeddings, use_cuda, ELMo_model_path, epochs_count, use_descriptions, descriptions_path
from gibber.gibberish_estimator import GibberishEstimator
from gibber.mixed_dataset import MixedDataset

if use_descriptions:
    import gibber.parsing # runs Morfeusz analyzer in background

if ELMo_model_path and POS_extended_model:
    raise Exception('Cannot combine ELMo with a POS-extended model.')

#
# POS groups for constructing extended lemmas when using a POS-aware model.
#
# If POS_extended_model is set to True, when training the model all lemmas will be
# converted to their extended form, and during prediction -- only when they're being
# fed to the model.
pos_groups = { 'subst': 'S', 'depr': 'S', 'brev': 'S',
               'fin': 'V', 'bedzie': 'V', 'aglt': 'V', 'praet': 'V', 'impt': 'V', 'imps': 'V', 'inf': 'V', 'pcon': 'V', 'pant': 'V', 'ger': 'V', 'pact': 'V', 'ppas': 'V', 'winien': 'V',
               'adj': 'A', 'adja': 'A', 'adjp': 'A', 'adjc': 'A'
        }
def POS_extended_lemma(lemma, tag):
    tag_pos = tag.split(':')[0]
    if tag_pos in pos_groups:
        return lemma+':'+pos_groups[tag_pos]
    return lemma


def words_window(sent_tokens, tid, replace_target = False):
    """replace_target can contain a word to replace the token at tid with."""
    return ([''] * -min(tid-window_size, 0)
            + sent_tokens[max(tid-window_size, 0) : tid]
            + [ (replace_target if replace_target else sent_tokens[tid]) ]
            + sent_tokens[tid+1 : tid+1+window_size]
            + [''] * (tid+window_size-len(sent_tokens)+1)
            )

#
## Get word vectors
#

if ELMo_model_path:
    import elmoformanylangs
    logging.disable(logging.WARNING) # stop ELMo for printing out loads of stuff to the console
    embedding = elmoformanylangs.Embedder(ELMo_model_path)
    print('ELMo model loaded from {}'.format(ELMo_model_path))
else:
    from gibber import word_embeddings
    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embeddings.word_vecs), freeze=freeze_embeddings)
    from gibber.word_embeddings import word_id, bound_token_id, vecs_count

#
## Load or train the model for estimating 'gibberish'.
#
model = GibberishEstimator(embedding, lstm_hidden_size, lstm_layers_count, lstm_is_bidirectional, use_cuda=use_cuda, use_elmo=(True if ELMo_model_path else False))
if use_cuda:
    model = model.cuda()

optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=reg_rate)
loss = torch.nn.BCELoss() # binary cross-entropy

if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
    print('A pretrained model loaded from {}'.format(model_path))
else:
    print('No pretrained model found in {}, training anew'.format(model_path))
    ## Read the NKJP fragments, as lists of lemmas
    words_count = 0
    train_sents = []

    if nkjp_format == 'plain':
        if POS_extended_model:
            raise ValueError('Cannot train a POS-extended model from plain corpus format.')
        with open(nkjp_path) as corp_file:
            for line in corp_file:
                lemmas = line.strip().split()
                train_sents.append([l.lower() for l in lemmas])
                words_count += len(lemmas)
    if nkjp_format == 'corpus':
        with open(nkjp_index_path) as index:
            for fragm_id in index:
                fragm_id = fragm_id.strip()
                # tag is namespaced, .// for finding anywhere in the tree
                tokens_filepath = nkjp_path+fragm_id+'/ann_morphosyntax.xml'
                if not os.path.isfile(tokens_filepath):
                    print('Note: cannot access {}'.format(fragm_id.strip()))
                    continue
                tree = etree.parse(tokens_filepath)
                for sent_subtree in tree.iterfind('.//{http://www.tei-c.org/ns/1.0}s[@{http://www.w3.org/XML/1998/namespace}id]'):
                    sent_lemmas = []
                    for seg in sent_subtree.iterfind('.//{http://www.tei-c.org/ns/1.0}f[@name]'):
                        if seg['name'] != 'disamb':
                            continue
                        interp = seg.find('.//{http://www.tei-c.org/ns/1.0}string').text.split(':')
                        lemma = interp[0].lower()
                        tag = ':'.join(interp[1:])
                        if POS_extended_model:
                            lemma = POS_extended_lemma(lemma, tag)
                        sent_lemmas.append(lemma)
                    train_sents += [[]]
                    for word in sent_lemmas:
                        train_sents[-1].append(word)
                        words_count += 1

    ### Torch - training the "language/gibberish discriminator"

    ##
    ## Training corpus preparation.
    ##
    ## We want a training set of multiword contexts (train) and target words (labels);
    ## the model will learn to distinguish genuine from fake (negative) context-target pairs
    batch_size = 128
    if ELMo_model_path: # corpus preparation for ELMo case, as lemma strings
        sample_n = 0
        train = []
        labels = np.ones(((words_count + words_count) * corp_runs,),
                        dtype='int')

        for run_n in range(corp_runs):
            sent_n = 0
            word_n = 0
                
            while sent_n < len(train_sents) and sample_n < labels.shape[0]:
                # The positive sample.
                train.append(words_window(train_sents[sent_n], word_n))
                # (the "one" positive label is the default)
                # labels[sample_n] = 1.0
                
                # The negative sample.
                sample_n += 1
                neg_sample = train[-1][:] # make a copy
                shuffle(neg_sample)
                # (replace two random words)
                neg_sample[randint(0, len(neg_sample)-1)] = choice(choice(train_sents))
                neg_sample[randint(0, len(neg_sample)-1)] = choice(choice(train_sents))
                train.append(neg_sample)
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

        train_set = MixedDataset(batch_size, train, torch.Tensor(labels))

    else: # corpus preparation for non-ELMo case, as word vector indices
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

        train_set = MixedDataset(batch_size, torch.LongTensor(train), torch.Tensor(labels))

    for epoch_n in range(epochs_count):
        for i, batch in enumerate(train_set):
            samples, labels = batch
            if use_cuda:
                labels = labels.cuda() # the model will take care of samples, which may be string lists for ELMo
            if not ELMo_model_path:
                samples = Variable(samples)
            labels = Variable(labels).view(labels.size(0), 1)
            predictions = model(samples)
            loss_val = loss(predictions, labels)
            loss_val.backward()
            optimizer.step()
            if i % 50 == 0:
                print('Loss at batch {}: {}'.format(i, float(loss_val.item())))

    torch.save(model.state_dict(), model_path)
    print('New model weights saved.')

    # cleanup to save some memory
    train, labels = None, None

##
## Dynamic storage of word relations info
##
wordnet_xml = etree.parse(pl_wordnet_path)

# Relations we're interested in:
# only hyper/hyponymy, similarity, synonymy
rels1 = { '11', '60', '30' }
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

# POS equivalences for mapping NKJP tagset to Wordnet POS.
pos_equivalences = {
        'czasownik': [ 'fin', 'bedzie', 'aglt', 'praet', 'impt', 'imps', 'inf', 'pcon', 'pant', 'ger', 'pact', 'ppas', 'winien', 'pred', 'qub', 'brev' ],
        'przymiotnik': [ 'adj', 'adja', 'adjp', 'adjc', 'qub', 'brev' ],
        'przysłówek': [ 'adv', 'qub', 'brev' ],
        'rzeczownik': [ 'subst', 'depr', 'ppron12', 'ppron3', 'qub', 'brev' ]
        }

# NOTE. Theoretically, we can have lex_unit ids and synsets given by annotation, but we want
# to take into account all possibilities given by raw word forms. Our model will try to
# **select** the correct lex_unit ids for given form in given context.
# NOTE 2. We store the bulk of relations in x1 dicts, x2 and x3 store only ones specific to them
# NOTE 3. "lexical units" and "wordids" in fact relate to distinct senses, as per the Wordnet

skl_word_wordids = defaultdict(list) # word -> ids as list
# /\ These are only wordids, full symbols are stored in the dict below!

# NOTE 4. We append POS strings and variant numbers to wordids, as in aaa_POS_111 (optionally
# just aaa_111 should be equivalent)
skl_wordid_symbols = dict()

skl_wordid_neighbors1 = defaultdict(list) # lemmas
skl_wordid_neighbors2 = defaultdict(list)
skl_wordid_neighbors3 = defaultdict(list)

skl_wordid_descs = dict() # here we store raw desc strings if we use them (see config)

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
def normalize_lemma(l, tag):
    if l in persons:
        return 'osoba'
    return l

def add_word_neighborhoods(words):
    """Retrieve neighbor senses from Wordnet for all given (lemma, tag) pairs and store them
    in internal representation for later prediction."""

    norm_words = set()
    for (w, t) in words:
        try:
            norm_words.add((normalize_lemma(w, t), t))
        except ValueError as ve:
            print(ve)
            norm_words.add((w, t))
    # NOTE Here we remove tags.
    words = [w for (w, t) in norm_words]

    new_words = set([w for w in words if not w in skl_word_wordids])

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
            if form in new_words or rm_sie(form) in new_words:
                skl_word_wordids[form].append(lex_unit.get('id'))
                # Preserve the variant number in full symbol.
                skl_wordid_symbols[lex_unit.get('id')] = (lex_unit.get('id')
                                                          +'_'+lex_unit.get('pos')
                                                          +'_'+lex_unit.get('variant'))
                if use_descriptions:
                    skl_wordid_descs[lex_unit.get('id')] = lex_unit.get('desc')

        # The r.* dicts are obtained by reversal of their parent dictionary when needed by the neighborhood function.
        # (reverse dict)
        rskl_word_wordids = reverse_list_dict(skl_word_wordids) # ids -> words

        # !!! From now on, make sure that this dict only contains new_words.
        # (note that rskl_word_wordids gives us singleton lists as dict values)
        rskl_word_wordids = dict([(id, words) for (id, words) in rskl_word_wordids.items() if words[0] in new_words])

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
        for synrel in wordnet_xml.iterfind('synsetrelations'):
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
        rskl_wordid_neighbor_ids2 = reverse_list_dict(skl_wordid_neighbor_ids2)
        rskl_wordid_neighbor_ids3 = reverse_list_dict(skl_wordid_neighbor_ids3)

        # Finally, harness neighbors by their known wordids.
        for lex_unit in wordnet_xml.iterfind('lexical-unit'):
            if lex_unit.get('id') in rskl_wordid_neighbor_ids1:
                for receiver in rskl_wordid_neighbor_ids1[lex_unit.get('id')]: # words "interested" in this neighbor
                    skl_wordid_neighbors1[receiver].append(lex_unit.get('name').lower())
            if lex_unit.get('id') in rskl_wordid_neighbor_ids2:
                for receiver in rskl_wordid_neighbor_ids2[lex_unit.get('id')]:
                    skl_wordid_neighbors2[receiver].append(lex_unit.get('name').lower())
            if lex_unit.get('id') in rskl_wordid_neighbor_ids3:
                for receiver in rskl_wordid_neighbor_ids3[lex_unit.get('id')]:
                    skl_wordid_neighbors3[receiver].append(lex_unit.get('name').lower())

##
## Predicting senses.
##

sample = np.zeros((1, window_size * 2 + 1), dtype='int')
def fill_sample(lemma, sent, token_id):
    # Fill with word ids which will be replaced with vectors by the embedding.
    sample[0, window_size] = word_id(lemma) # the target word
    for j in range(window_size): # its neighbors in the window_size
        sample[0, j] = (word_id(sent[token_id-j-1])
                        if token_id-j-1 >= 0
                        else bound_token_id)
        sample[0, window_size+j+1] = (word_id(sent[token_id+j+1])
                                        if token_id+j+1 < len(sent)
                                        else bound_token_id)

def normalize_and_decide(sense_probs, sense_wordids, verbose=False):
    # Get normalized probabilities.
    normal_sum = sum(sense_probs)
    sense_normd_probs = []
    for prob in sense_probs:
        sense_normd_probs.append(prob/normal_sum)
        
    winners = [sid for (sid, p) in enumerate(sense_normd_probs) if p == max(sense_normd_probs)]
    winner_id = choice(winners)
    predicted_sense = skl_wordid_symbols[sense_wordids[winner_id]]
    if verbose:
        print('Computed probs are {}'.format(sense_normd_probs))
        print('The winner is {} ({})'.format(predicted_sense, sense_normd_probs[winner_id]))

    return winner_id, predicted_sense, sense_normd_probs

def predict_with_subset(sent, token_id, subset_name, sense_wordids, sense_ngbs, sense_probs,
                          fallback_probs=False, fallback_ngbcounts=False, verbose=False):
    if verbose:
        print('#', subset_name)
    sense_ngbcounts = [len(s) for s in sense_ngbs]
    senses_considered = 0
    for (sid, ngbs) in enumerate(sense_ngbs):
        if len(ngbs) == 0: # if we have no specific info on this set of relations
            if fallback_probs and fallback_ngbcounts[sid] > 1: # just carry the fallback average
                sense_probs[sid] = fallback_probs[sid]
                senses_considered += 1 # it also counts as "considered"
            continue
        senses_considered += 1
        if ELMo_model_path:
            # Send ngb_words in bulk to the model and then populate sense_probs.
            probs = model([words_window(sent, token_id,
                                       replace_target=ngb_word)
                                       for ngb_word in ngbs]).detach().cpu().numpy()
            sense_probs[sid] = float(np.mean(probs))
        else:
            for ngb_word in ngbs:
                if POS_extended_model: # assume the same tag/POS as target
                    fill_sample(POS_extended_lemma(ngb_word, tag), sent, token_id)
                else:
                    fill_sample(ngb_word, sent, token_id)
                sense_probs[sid] += model(torch.LongTensor(sample)).detach().cpu()

            # We have average prob of any neighbor of this sense
            sense_probs[sid] /= sense_ngbcounts[sid]
        # We have average prob of any neighbor of this sense + carry the input from the fallback
        # (note that it can be ignored due to zero neighbors)
        sense_probs[sid] = (sense_probs[sid] * sense_ngbcounts[sid]
                                    / sense_ngbcounts[sid])
        if fallback_probs:
            sense_probs[sid] += (fallback_probs[sid] * fallback_ngbcounts[sid]
                                    / fallback_ngbcounts[sid])

        if verbose:
            print('* Sense {}: {}'.format(skl_wordid_symbols[sense_wordids[sid]], ngbs))
    winner_id, predicted_sense, sense_normd_probs = normalize_and_decide(sense_probs, sense_wordids)
    decision = (predicted_sense, sense_normd_probs[winner_id], len(sense_wordids), senses_considered)
    return decision

def merge_predictions(subset_name, sense_wordids, reference_prob, ngbcounts, probs, verbose=False):
    """Both ngbcounts and probs should be aligned lists of relevant objects from predictions that
    are to be merged."""
    if verbose:
        print('#', subset_name)
    senses_considered = 0
    merged_probs = [reference_prob for p in probs[0]]
    for (sense_n, _) in enumerate(ngbcounts[0]):
        counts = [pred_counts[sense_n] for pred_counts in ngbcounts]
        if any([count > 0 for count in counts]):
            senses_considered += 1
            merged_probs[sense_n] = 0.0
            for (pred_n, count) in enumerate(counts):
                merged_probs[sense_n] += count * probs[pred_n][sense_n]
    winner_id, predicted_sense, sense_normd_probs = normalize_and_decide(merged_probs, sense_wordids)
    decision = (predicted_sense, sense_normd_probs[winner_id], len(sense_wordids), senses_considered)
    return decision

def predict_sense(token_lemma, tag, sent, token_id, verbose=False, discriminate_POS=True):
    """Get token_lemma and tag as strings, the whole sent as a sequence of strings, and token_id
    indicating the index where the token is in the sentence. Return decisions made when using
    subsets 1, 2, 3, 4 of relations as tuples (estimated probability, sense, retrieved_sense_count,
    considered_sense_count) or None's if no decision. Retrieved_sense_count indicates how many
    senses were found for the lemma, and considered_sense_count for how many we were able to find
    a neighborhood with the given subset of relations. If discriminate_POS is set, only senses
    of Wordnet POS matching the provided tag will be considered."""

    if ELMo_model_path:
        reference_prob = model([words_window(sent, token_id)]).detach().cpu()
    else:
        if POS_extended_model:
            fill_sample(POS_extended_lemma(token_lemma, tag), sent, token_id)
        else:
            fill_sample(token_lemma, sent, token_id)
        reference_prob = model(torch.LongTensor(sample)).detach().cpu() # probability of the sentence fragment with just the original token in center

    # For the purpose of finding neighbors, we need to do the adjustments.
    try:
        token_lemma = normalize_lemma(token_lemma, tag)
    except ValueError as ve:
        print(ve)

    sense_wordids = skl_word_wordids[token_lemma]
    if discriminate_POS:
        matching_wordids = []
        tagset_pos = tag.split(':')[0]
        for wordid in sense_wordids:
            wordnet_pos = skl_wordid_symbols[wordid].split('_')[1]
            if wordnet_pos in pos_equivalences and tagset_pos in pos_equivalences[wordnet_pos]:
                matching_wordids.append(wordid)
        sense_wordids = matching_wordids

    if len(sense_wordids) == 0:
        sense_wordids = skl_word_wordids[token_lemma + ' się']
        if len(sense_wordids) == 0:
            raise LookupError('no senses found for {}'.format(token_lemma))

    # Neighbors of each sense. A sense can have 0 neighbors in some relation set.
    sense_ngbs1 = [set(skl_wordid_neighbors1[swi]) for swi in sense_wordids]
    # Number of neighbors for each sense - important when calculating average for later relation subsets.
    sense_ngbcounts1 = [len(set(skl_wordid_neighbors1[swi])) for swi in sense_wordids]
    sense_probs1 = [reference_prob if (c == 0) else 0.0
                        for (x, c) in zip(sense_wordids, sense_ngbcounts1)]
    sense_ngbs2 = [set(skl_wordid_neighbors2[swi]) for swi in sense_wordids]
    sense_ngbcounts2 = [len(set(skl_wordid_neighbors2[swi])) for swi in sense_wordids]
    sense_probs2 = [reference_prob if (c == 0 and c1 == 0) else 0.0
                        for (x, c, c1) in zip(sense_wordids, sense_ngbcounts2, sense_ngbcounts1)]
    sense_ngbs3 = [set(skl_wordid_neighbors3[swi]) for swi in sense_wordids]
    sense_ngbcounts3 = [len(set(skl_wordid_neighbors3[swi])) for swi in sense_wordids]
    sense_probs3 = [reference_prob if (c == 0 and c1 == 0) else 0.0
                        for (x, c, c1) in zip(sense_wordids, sense_ngbcounts3, sense_ngbcounts1)]
    
    decision1, decision2, decision3, decision4 = None, None, None, None
    
    if verbose:
        print('\n# Disambiguating {} in {}'.format(token_lemma, sent))

    # 1st subset:
    if sum([len(x) for x in sense_ngbs1]) > 0: # we need a neighbor for at least one sense
        decision1 = predict_with_subset(sent, token_id, 'Relation subset 1', sense_wordids, sense_ngbs1, sense_probs1,
                                            verbose=verbose)

    # 2rd subset (1+2)
    if sum([len(x) for x in sense_ngbs2]) > 0: # there's a neighbor for at least one sense
        decision2 = predict_with_subset(sent, token_id, 'Relation subset 2', sense_wordids, sense_ngbs2, sense_probs2,
                                        fallback_probs=sense_probs1, fallback_ngbcounts=sense_ngbcounts1, verbose=verbose)

    # 3rd subset (1+3)
    if sum([len(x) for x in sense_ngbs3]) > 0: # there's a neighbor for at least one sense
        decision3 = predict_with_subset(sent, token_id, 'Relation subset 3', sense_wordids, sense_ngbs3, sense_probs3,
                                        fallback_probs=sense_probs1, fallback_ngbcounts=sense_ngbcounts1, verbose=verbose)
            
    # 4th subset (1+2+3)
    decision4 = merge_predictions('Relation subset 4', sense_wordids, reference_prob,
                                [sense_ngbcounts1, sense_ngbcounts2, sense_ngbcounts3],
                                [sense_probs1, sense_probs2, sense_probs3], verbose=verbose)

    # Optional run with descriptions.
    if use_descriptions:
        sense_probs5 = [reference_prob for s in sense_wordids]
        sense_descwords = [[] for s in sense_wordids]

        for sense_n, wordid in enumerate(sense_wordids):
            if descriptions_path:
                desc_lemmas = []
                expected_path = descriptions_path+'/'+wordid+'.txt'
                if os.path.isfile(expected_path):
                    with open(expected_path) as descs_file:
                        parsed_sents = json.loads(descs_file.read())
                        for parsed_sent in parsed_sents:
                            desc_lemmas += [ lemma for (form, lemma, tag) in parsed_sent ]
            else:
                samples_str = gibber.parsing.extract_samples(skl_wordid_descs[wordid]).strip()
                if len(samples_str) == 0:
                    continue
                desc_tokens = chain.from_iterable(gibber.parsing.parse_sentences(samples_str)) # merge sentences
                desc_lemmas = [lemma for (form, lemma, interp) in desc_tokens]
            sense_descwords[sense_n] = set(desc_lemmas)
        sense_ngbcounts5 = [len(set(sense_descwords[swi])) for swi, _ in enumerate(sense_wordids)]
        decision5 = predict_with_subset(sent, token_id, 'Unit descriptions', sense_wordids, sense_descwords,
                                            sense_probs5, verbose=verbose)
        decision6 = merge_predictions('Unit descriptions + relations', sense_wordids, reference_prob,
                                    [sense_ngbcounts1, sense_ngbcounts2, sense_ngbcounts3, sense_ngbcounts5],
                                    [sense_probs1, sense_probs2, sense_probs3, sense_probs5], verbose=verbose)

        return (decision1, decision2, decision3, decision4, decision5, decision6)

    return (decision1, decision2, decision3, decision4)

def random_prediction(token_lemma, tag, verbose=False, discriminate_POS=True):
    # For the purpose of finding neighbors, we need to do the adjustments.
    try:
        token_lemma = normalize_lemma(token_lemma, tag)
    except ValueError as ve:
        print(ve)

    sense_wordids = skl_word_wordids[token_lemma]
    ###sense_symbols = [skl_wordid_symbols[wid] for wid in sense_wordids]
    ###print(token_lemma, tag, sense_symbols)
    if discriminate_POS:
        matching_wordids = []
        tagset_pos = tag.split(':')[0]
        for wordid in sense_wordids:
            wordnet_pos = skl_wordid_symbols[wordid].split('_')[1]
            if wordnet_pos in pos_equivalences and tagset_pos in pos_equivalences[wordnet_pos]:
                matching_wordids.append(wordid)
        sense_wordids = matching_wordids

    if len(sense_wordids) == 0:
        sense_wordids = skl_word_wordids[token_lemma + ' się']
        if len(sense_wordids) == 0:
            raise LookupError('no senses found for {}'.format(token_lemma))

    sense_symbols = [skl_wordid_symbols[wid] for wid in sense_wordids]
    winner = choice(sense_symbols)
    if verbose:
        print('Among {} the random winner is {}'.format(sense_symbols, winner))
    return winner

def first_variant_prediction(token_lemma, tag, verbose=False, discriminate_POS=True):
    # For the purpose of finding neighbors, we need to do the adjustments.
    try:
        token_lemma = normalize_lemma(token_lemma, tag)
    except ValueError as ve:
        print(ve)

    sense_wordids = skl_word_wordids[token_lemma]
    if discriminate_POS:
        matching_wordids = []
        tagset_pos = tag.split(':')[0]
        for wordid in sense_wordids:
            wordnet_pos = skl_wordid_symbols[wordid].split('_')[1]
            if wordnet_pos in pos_equivalences and tagset_pos in pos_equivalences[wordnet_pos]:
                matching_wordids.append(wordid)
        sense_wordids = matching_wordids

    if len(sense_wordids) == 0:
        sense_wordids = skl_word_wordids[token_lemma + ' się']
        if len(sense_wordids) == 0:
            raise LookupError('no senses found for {}'.format(token_lemma))

    sense_symbols = [skl_wordid_symbols[wid] for wid in sense_wordids]
    winner = None
    for symb in sense_symbols:
        if re.search('_1$', symb):
            winner = symb
    if verbose:
        print('Among {} the first-variant winner is {}'.format(sense_symbols, winner))
    return winner

def sense_match(full, partial):
    full = full.split('_')
    lexid, variant = full[0], full[2]
    return re.match('^({})?_([^_]+_)?({})?$'.format(lexid, variant), partial) is not None
