import os, csv, re
from operator import itemgetter
from lxml import etree

def load_skladnica_wn2(skladnica_path, skladnica_sections_index_path):
    sents = [] # pairs: (word lemma, lexical unit id [or None])
    words = set() # all unique words that are present
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
                                words.add((lemma, tag))
                        sent.sort(key=itemgetter(0))
                        sent = [(token, lexical_id, tag) for num, token, lexical_id, tag in sent]
                        sents.append(sent)
    return sents, words

def load_wn3_corpus(annot_sentences_path):
    sents = [] # pairs: (word lemma, lexical unit id [or None])
    words = set() # all unique words that are present

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

    return sents, words
