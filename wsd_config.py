#
# gibber_wsd configuration
#

nkjp_format = "plain" # "plain" - space separated lemmas one sentence per line, "corpus" - original NKJP format (as tree of directories)
nkjp_index_path = "nkjp_index.txt" # only when "corpus"
nkjp_path = './NKJP-1M/NKJP1M.txt'
#nkjp_path = "/home/szymon/lingwy/nkjp/pełny/" # directory when "corpus", else a text file

vecs_path = "/home/szymon/lingwy/nkjp/wektory/nkjp+wiki-lemmas-all-300-cbow-ns-50.txt"
#pl_wordnet_path = "/home/szymon/lingwy/wielozn/wzięte/plwordnet_2_0_4/plwordnet_2_0_4_release/plwordnet_2_0.xml"
pl_wordnet_path = '/home/szymon/lingwy/wielozn/plwordnet_3_1/plwordnet-3.1.xml'

# model will be either trained and saved to, or loaded from this file (if exists)
model_path = '../WSD-rew/wsd-nkjp300-weights.h5'
#model_path = '../WSD-rew/wsd-nkjp1m-weights.h5'

use_pos = False # whether to use POS information for the gibberish estimator (not implemented!)

#
# (Experim configuration, ignored by gibber_wsd):
#
#mode = 'wordnet2_annot'
mode = 'wordnet3_annot'

transform_lemmas = False # transform lemmas of gerunds from infinitive to gerund form?

full_diagnostics = False # print detailed diagnostics for each prediction case?
diagnostics_when_23_fails = False # show diagnostics for cases when subset 2 or 3 fails and the 1st is correct

# If this is not None, four predictions will be printed to CSV files, containing in their
# columns: lemma, tag, sense variant number, lexical id (ie. Wordnet unit identifier)
output_prefix = 'wsd_prediction_'

# only for wordnet2_annot mode
skladnica_sections_index_path = "/home/szymon/lingwy/nkjp/skladnica_znacz/sections.txt"
skladnica_path = "/home/szymon/lingwy/nkjp/skladnica_znacz/"

# only for wordnet3_annot mode
annot_sentences_path = 'oznaczenia-wszystko-plwordnet-gotowe-new.csv'
