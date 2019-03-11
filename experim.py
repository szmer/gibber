from wsd_config import nkjp_index_path, mode, skladnica_sections_index_path, skladnica_path, annot_sentences_path, output_prefix, full_diagnostics, diagnostics_when_23_fails, pl_wordnet_path, use_descriptions 
from gibber.wsd import add_word_neighborhoods, predict_sense, sense_match
from gibber.annot_corp_loader import load_skladnica_wn2, load_wn3_corpus

print('mode: {}\nNKJP: {}\nWordnet: {}'.format(mode, nkjp_index_path, pl_wordnet_path))

### Get Składnica sentences

# NOTE this also contains empty lists where the sents were actually not annotated.
sents = [] # pairs: (word lemma, lexical unit id [or None])
words = set() # all unique tagged words that are present

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

class ResultCategory(object):
    def __init__(self, fallback_cats=[]):
        self.fallback_cats = fallback_cats # used for words_checked unique words counting

        self.num_some_ngbs = 0
        self.num_all_ngbs = 0
        self.good_some_ngbs = 0
        self.good_all_ngbs = 0

        self.unique_good_some_ngbs = 0
        self.unique_good_all_ngbs = 0
        self.words_checked = set() # for determining uniqueness, only words with 'all' coverage

    def print_scores(self, num_all, num_all_unique):
        print('All: {}, good: {}, accuracy: {}'.format(num_all, self.good_some_ngbs,
                (self.good_some_ngbs/num_all if num_all > 0 else 'none')))
        print('Unique: {}, good: {}, accuracy: {}'.format(num_all_unique, self.unique_good_some_ngbs,
                (self.unique_good_some_ngbs/num_all_unique if num_all_unique > 0 else 'none')))
        print('With full ngb coverage: {}, good: {}, accuracy: {}'.format(self.num_all_ngbs, self.good_all_ngbs,
                (self.good_all_ngbs/self.num_all_ngbs if self.num_all_ngbs > 0 else 'none')))
        full_unique_counts = len(self.words_checked) + sum([len(c.words_checked) for c in self.fallback_cats])
        print('Unique w/full coverage: {}, good: {}, accuracy: {}'.format(len(self.words_checked), self.unique_good_all_ngbs,
                (self.unique_good_all_ngbs/full_unique_counts if full_unique_counts > 0 else 'none')))

## numbers indicate the relation group,
## A - words where at least one sense has known neighbors,
## B - words where all senses have known neighbors,
## uq - no word repetitions
## "Good" values will be divided by N to get accuracy.
words_checked_some_senses = set() # for determining uniqueness
words_checked_rest = set() # for calculating the total performance
num_all = 0 # number of all occurences where we have some true sense provided

relations1 = ResultCategory()
relations2 = ResultCategory(fallback_cats=[relations1])
relations3 = ResultCategory(fallback_cats=[relations1])
relations4 = ResultCategory(fallback_cats=[relations1, relations2, relations3])
unitdescs = ResultCategory()
unitdescs_withrels = ResultCategory(fallback_cats=[relations1, relations2, relations3])

##
## Collect model decisions for all subsets of relations, counting correct decisions.
##

if output_prefix:
    out1 = open(output_prefix+'1', 'w+')
    out2 = open(output_prefix+'2', 'w+')
    out3 = open(output_prefix+'3', 'w+')
    out4 = open(output_prefix+'4', 'w+')
    if use_descriptions:
        outd = open(output_prefix+'5', 'w+')
        outdr = open(output_prefix+'6', 'w+')
else:
    out1 = False
    out2 = False
    out3 = False
    out4 = False
    outd = False
    outdr = False

def rate_decision(decision, result_cat, new_word, output_file=False, fallback_decision=None, fallback_wordsets=[]):
    """Return a boolean, whether the decision was correct."""
    good = False
    if decision is not None:
        sense_id, prob, senses_count, considered_sense_count = decision
    elif fallback_decision is not None:
        sense_id, prob, senses_count, considered_sense_count = fallback_decision
    else:
        if output_file:
            print('{},{},{},{}'.format(lemma, tag, '?', '?'), file=output_file)
        return good

    if output_file:
        sense_data = sense_id.split('_')
        print('{},{},{},{}'.format(lemma, tag, sense_data[2], sense_data[0]), file=output_file)
    # Increase occurence counts.
    result_cat.num_some_ngbs += 1
    if senses_count == considered_sense_count:
        result_cat.num_all_ngbs += 1
    # Correct?
    if sense_match(sense_id, true_sense):
        result_cat.good_some_ngbs += 1
        good = True
        if senses_count == considered_sense_count:
            result_cat.good_all_ngbs += 1
    # Unique word indication.
    if new_word:
        if senses_count == considered_sense_count and not any(((lemma in ws) for ws in fallback_wordsets)):
            result_cat.words_checked.add(lemma) # marked as checked with all info on all senses

        if sense_match(sense_id, true_sense):
            result_cat.unique_good_some_ngbs += 1
            if senses_count == considered_sense_count:
                result_cat.unique_good_all_ngbs += 1
    return good

print('Performing the test on sense-annotated sentences.')
for (sid, sent) in enumerate(sents):
    print('Testing sentence', sid+1)
    for (tid, token_data) in enumerate(sent):
        lemma, true_sense, tag = token_data[0], token_data[1], token_data[2]
        if true_sense is None:
            if output_prefix:
                print('{},{},{},{}'.format(lemma, tag, '<<<', '<<<'), file=out1)
                print('{},{},{},{}'.format(lemma, tag, '<<<', '<<<'), file=out2)
                print('{},{},{},{}'.format(lemma, tag, '<<<', '<<<'), file=out3)
                print('{},{},{},{}'.format(lemma, tag, '<<<', '<<<'), file=out4)
                if use_descriptions:
                    print('{},{},{},{}'.format(lemma, tag, '<<<', '<<<'), file=outd)
                    print('{},{},{},{}'.format(lemma, tag, '<<<', '<<<'), file=outdr)
            continue

        num_all += 1

        try:
            if use_descriptions:
                decision1, decision2, decision3, decision4, decision5, decision6 = predict_sense(lemma, tag,
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
        if not lemma in words_checked_some_senses:
            words_checked_some_senses.add(lemma)
            new_word = True

        # preserve an indication if the first decision was good to show diagnostics when 3 or 3 fails
        good1 = rate_decision(decision1, relations1, new_word, output_file=out1)
        good2 = rate_decision(decision2, relations2, new_word, output_file=out2, fallback_decision=decision1, fallback_wordsets=[relations1.words_checked])
        good3 = rate_decision(decision3, relations3, new_word, output_file=out3, fallback_decision=decision1, fallback_wordsets=[relations1.words_checked])
        rate_decision(decision4, relations4, new_word, output_file=out4, fallback_wordsets=[relations1.words_checked, relations2.words_checked, relations3.words_checked])
        if use_descriptions:
            rate_decision(decision5, unitdescs, new_word, output_file=outd)
            rate_decision(decision6, unitdescs_withrels, new_word, output_file=outdr, fallback_wordsets=[relations1.words_checked, relations2.words_checked, relations3.words_checked])

        # Re-run with diagnostics when diagnostics_when_23_fails condition is true.
        if diagnostics_when_23_fails and (not full_diagnostics) and good1 and not (good2 and good3):
            predict_sense(lemma, tag, [tok_info[0] for tok_info in sent], # give only lemmas
                    tid, verbose=True)
            print('The true sense was', true_sense)

if output_prefix:
    out1.close()
    out2.close()
    out3.close()
    out4.close()
    if use_descriptions:
        outd.close()
        outdr.close()

num_all_unique = len(words_checked_some_senses)+len(words_checked_rest) # "rest" only captures fundamental prediction failure

print('\nRelations subset 1')
relations1.print_scores(num_all, num_all_unique)
print('\nRelations subset 2')
relations2.print_scores(num_all, num_all_unique)
print('\nRelations subset 3')
relations3.print_scores(num_all, num_all_unique)
print('\nRelations subset 4')
relations4.print_scores(num_all, num_all_unique)
if use_descriptions:
    print('\nUnit descriptions')
    unitdescs.print_scores(num_all, num_all_unique)
    print('\nUnit descriptions+relations.')
    unitdescs_withrels.print_scores(num_all, num_all_unique)
