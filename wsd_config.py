##
## gibber package configuration
##

# Shold the language model use forms instead of lemmas?
# Works only with models trained by this package to deal specifically with forms.
use_forms = False

# Shold the tokens used by language model be part of speech-tagged?
# Works only with appropriate vector embeddings AND models trained by this package to deal specifically with POS-tagged tokens.
POS_extended_model = False

# How probabilities of sentence from neighbors in one set should be integrated?
probability_collection = 'average' # max or average

# Include Wordnet unit descriptions in testing?
# Needs either Concraft or preparsed lexical units (see paths config).
use_descriptions = True

# Give an additional 'voted' prediction (relations1, relations4, descriptions) - descriptions must be used!
give_voted_pred = False

# Use CUDA GPU capability if possible?
use_cuda = True

#
# Paths to essential resources.
#

# Word vectors (a file in FastText-like format - word and space-separated numbers in each row), likely downloaded from http://dsmodels.nlp.ipipan.waw.pl
# Meaningless when using ELMo.
vecs_path = "../WSD-dostane/nkjp+wiki-lemmas-all-300-cbow-ns-50.txt"
#vecs_path = "/home/szymon/lingwy/nkjp/wektory/nkjp+wiki-forms-all-300-cbow-ns-50.txt"
#vecs_path = "../WSD-new/nkjp+wiki-lemmas-pos-cbow-ns-50.txt"

# Path to PlWordnet (XML version).
pl_wordnet_path = "/home/szymon/lingwy/wielozn/wzięte/plwordnet_2_0_4/plwordnet_2_0_4_release/plwordnet_2_0.xml"
#pl_wordnet_path = "/home/szymon/lingwy/wielozn/plwordnet_3_1/plwordnet-3.1.xml"

# If you don't use Gensim, new language model will be either trained and saved to, or loaded from this file (if exists)
model_path = 'models/nkjp1m_nopos.torch'
#model_path = 'models/nkjp1m_nopos_elmo.torch'
#model_path = 'models/nkjp1m_nopos_forms.torch'

# NKJP is needed if you want to train a model anew.
nkjp_format = "corpus" # "plain" - space separated lemmas one sentence per line, "corpus" - original NKJP format (as tree of directories)
nkjp_index_path = "nkjp_index.txt" # only when "corpus"
#nkjp_path = "./NKJP-1M/NKJP1M.txt"
nkjp_path = "/home/szymon/lingwy/nkjp/pełny/" # directory when "corpus", else a text file

# Path to a model saved from Gensim, if you want to use one (otherwise leave False).
# Not compatible with POS extension
gensim_model_path = False # '../gensim_Embeddings/nkjp+wiki-lemmas-all-300-skipg-hs'

# Path to ELMo model in elmoinmanylangs format.
# Can be False if you don't want to use ELMo. Not compatible with POS extension, elmoformanylangs package needed
ELMo_model_path = False #'/home/szymon/lingwy/elmo/pl_elmo'

# Provide path to a catalog with pre-parsed sense glosses/descriptions (got from the mass parsing script)
descriptions_path = './parsed_descs/'
# if not provided (and the config is set to use glosses), we will try to parse on the fly with Concraft (no caching, so unnecessarily slow!)
concraft_model_path = '/home/szymon/lingwy/concraft/model-04-09-2018.gz'

#
# Language model settings (when you don't use Gensim).
#

# Model training
# How many words to consider on each side of the target settings.
window_size = 4
# How many times to feed the whole corpus during training
corp_runs = 1
epochs_count = 3

learning_rate = 0.3
# Regularization
reg_rate = 0.005
lstm_hidden_size = 9
lstm_layers_count = 1
lstm_is_bidirectional = False
freeze_embeddings = True
# Original settings:
#reg_rate = 0.005
#lstm_hidden_size = 9
#lstm_layers_count = 1
#lstm_is_bidirectional = False
#freeze_embeddings = True

###############################################################################

##
## (Experim script configuration, ignored by gibber_wsd):
##

# Use one of the modes for different corpora (wordnet2_annot, wordnet3_annot, kpwr_annot)
mode = "wordnet3_annot"
#mode = "wordnet2_annot"
#mode = "kpwr_annot"

# If this is not False, four predictions will be printed to CSV files, containing in their
# columns: lemma, tag, sense variant number, lexical id (ie. Wordnet unit identifier)
output_prefix = 'gensim9_prediction_avg_'

# Print detailed diagnostics for each prediction case?
full_diagnostics = False
# Print diagnostics specifically for cases when subset 2 or 3 fails and the 1st is correct
diagnostics_when_23_fails = False

# only for wordnet2_annot mode
skladnica_sections_index_path = "/home/szymon/lingwy/nkjp/skladnica_znacz/sections.txt"
skladnica_path = "/home/szymon/lingwy/nkjp/skladnica_znacz/"

# only for wordnet3_annot mode
annot_sentences_path = "oznaczenia-wszystko-plwordnet-gotowe-new.csv"

# only for kpwr_annot mode
kpwr_path = "/home/szymon/lingwy/wielozn/kpwr2503"

###############################################################################

# baseline.py config, ignored by experim.py
# baseline type: "random" or "first-variant"
baseline = "random"

###############################################################################

##
## Mass description parsing config.
##

# Remember to set the correct mode for Experim to parse words for the right corpus!

# Parsing mode.
# 'corpus' (ie. what is needed for its annotation) or 'identifiers' - see below
mass_parsing_mode = 'identifiers'

# Here the parsed units will be saved
mass_parsing_path = './parsy2/'
# Parsing mode.
# 'corpus' (ie. everything that is needed for its annotation) or 'identifiers' - see below
mass_parsing_mode = 'identifiers'
# You can *copy* the file All_requested from the parses catalog and remove
# identifiers that were already done (useful when the script crashed)
mass_parsing_identifiers_file = 'remaining'
