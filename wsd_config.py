#
# gibber_wsd configuration
#

nkjp_format = "plain" # "plain" - space separated lemmas one sentence per line, "corpus" - original NKJP format (as tree of directories)
nkjp_index_path = "nkjp_index.txt" # only when "corpus"
nkjp_path = "./NKJP-1M/NKJP1M.txt"
#nkjp_path = "/home/szymon/lingwy/nkjp/pełny/" # directory when "corpus", else a text file

#vecs_path = "../WSD-dostane/nkjp+wiki-lemmas-all-300-cbow-ns-50.txt"
vecs_path = "../WSD-new/nkjp+wiki-lemmas-pos-cbow-ns-50.txt"
#pl_wordnet_path = "/home/szymon/lingwy/wielozn/wzięte/plwordnet_2_0_4/plwordnet_2_0_4_release/plwordnet_2_0.xml"
pl_wordnet_path = "/home/szymon/lingwy/wielozn/plwordnet_3_1/plwordnet-3.1.xml"

# model will be either trained and saved to, or loaded from this file (if exists)
model_path = 'nkjp1m_nopos_elmo.torch'
#model_path = "../WSD-rew/wsd-nkjp1m-pos-weights.h5"
#model_path = "../WSD-rew/wsd-nkjp300-pos-weights.h5"
#model_path = "../WSD-rew/wsd-nkjp300-weights.h5"
#model_path = "../WSD-rew/wsd-nkjp1m-weights.h5"

ELMo_model_path = '/home/szymon/lingwy/elmo/pl_elmo' # can be False if you don't want to use ELMo, not compatible with POS extension, elmoformanylangs package needed
POS_extended_model = False

# model training settings:
use_cuda = True
window_size = 4 # how many words to consider on both sides of the target
corp_runs = 2 # how many times to feed the whole corpus during training

learning_rate = 0.3
reg_rate = 0.005 # regularization
lstm_hidden_size = 9
lstm_layers_count = 1
lstm_is_bidirectional = False
freeze_embeddings = True
# Original settings:
#lstm_hidden_size = 9
#lstm_layers_count = 1
#lstm_is_bidirectional = False
#freeze_embeddings = True

#
# (Experim configuration, ignored by gibber_wsd):
#

#mode = "wordnet2_annot"
mode = "wordnet3_annot"

full_diagnostics = False # print detailed diagnostics for each prediction case?
diagnostics_when_23_fails = False # show diagnostics for cases when subset 2 or 3 fails and the 1st is correct

# baseline.py config, ignored by experim.py
baseline = "first-variant" # "random" or "first-variant"

# If this is not None, four predictions will be printed to CSV files, containing in their
# columns: lemma, tag, sense variant number, lexical id (ie. Wordnet unit identifier)
output_prefix = "wsd_prediction_"
baseline_output_prefix = "wsd_baseline"

# only for wordnet2_annot mode
skladnica_sections_index_path = "/home/szymon/lingwy/nkjp/skladnica_znacz/sections.txt"
skladnica_path = "/home/szymon/lingwy/nkjp/skladnica_znacz/"

# only for wordnet3_annot mode
annot_sentences_path = "oznaczenia-wszystko-plwordnet-gotowe-new.csv"
