nkjp_format = "corpus" # "plain" - space separated lemmas one sentence per line, "corpus" - original format
nkjp_index_path = "nkjp_index.txt" # only when "corpus"
nkjp_path = "/home/szymon/lingwy/nkjp/pełny/" # directory when "corpus", else a text file

vecs_path = "/home/szymon/lingwy/nkjp/wektory/nkjp+wiki-lemmas-all-100-skipg-ns.txt/data"
#pl_wordnet_path = "/home/szymon/lingwy/wielozn/wzięte/plwordnet_2_0_4/plwordnet_2_0_4_release/plwordnet_2_0.xml"
pl_wordnet_path = '/home/szymon/lingwy/wielozn/plwordnet_3_1/plwordnet-3.1.xml'

# (Experim configuration, ignored by gibber_wsd):
mode = 'wordnet2_annot'
#mode = 'wordnet3_annot'
transform_lemmas = False

# for wordnet2_annot mode
skladnica_sections_index_path = "/home/szymon/lingwy/nkjp/skladnica_znacz/sections.txt"
skladnica_path = "/home/szymon/lingwy/nkjp/skladnica_znacz/"

# for wordnet3_annot mode
annot_sentences_path = 'oznaczenia-wszystko-plwordnet3.csv'
