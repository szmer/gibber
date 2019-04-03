import glob, os, csv, re
from operator import itemgetter
from lxml import etree

def load_skladnica_wn2(skladnica_path, skladnica_sections_index_path):
    """Returns a list of sentences, as list of lemmas, and a set of words. The first has pairs: (form, lemma, true_sense, tag),
    the second: (lemma, tag)"""
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
                        
                        # Collect the list of sentence lemmas.
                        sent = []
                        for elem in tree.iterfind('node[@chosen]'):
                            if elem.attrib['chosen'] == 'true' and elem.find("terminal") is not None:
                                lexical_id = None
                                if elem.find('.//luident') is not None: # if word-sense annotation is available
                                    lexical_id = elem.find('.//luident').text+'_'

                                form = elem.find("terminal").find("orth").text
                                lemma = elem.find("terminal").find("base").text
                                if lemma is None: # happens for numbers
                                    lemma = form
                                lemma = lemma.lower()
                                tag = elem.find('.//f[@type]').text # should be <f type="tag">
                                sent.append((int(elem.attrib['from']), form, lemma, lexical_id, tag))
                                words.add((lemma, tag))
                        sent.sort(key=itemgetter(0))
                        sent = [(form, lemma, lexical_id, tag) for num, form, lemma, lexical_id, tag in sent]
                        sents.append(sent)
    return sents, words

def load_wn3_corpus(annot_sentences_path):
    """Returns a list of sentences, as list of lemmas, and a set of words. The first has pairs: (form, lemma, true_sense, tag),
    the second: (lemma, tag)"""
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
                    sent.append((form, lemma.lower(), '_'+true_sense, tag))
                else:
                    sent.append((form, lemma.lower(), None, tag))
                words.add((lemma.lower(), tag))

    return sents, words

def load_kpwr_corpus(kpwr_path):
    sents = []
    words = set()
    for xml_path in glob.glob(kpwr_path+'/*.xml'):
        if re.match('rel\.xml$', xml_path):
            continue
        with open(xml_path) as xml_file:
            xml = etree.parse(xml_file)
            for sent_obj in xml.iterfind('.//sentence'):
                sent = []
                senses_present = False
                for tok_obj in sent_obj.iterfind('tok'):
                    sense = False
                    for prop_obj in tok_obj.iterfind('prop'):
                        if prop_obj.attrib['key'] == 'wsd:sense:id':
                            sense = prop_obj.text
                            senses_present = True
                    form = tok_obj.find('orth').text
                    lex_obj = tok_obj.find('lex')
                    lemma = lex_obj.find('base').text.lower()
                    tag = lex_obj.find('ctag').text
                    words.add((lemma, tag))
                    if sense:
                        sent.append((form, lemma.lower(), sense+'_', tag)) # senses are given as lexical ids
                    else:
                        sent.append((form, lemma.lower(), None, tag))
                if senses_present and len(sent) > 0:
                    sents.append(sent)
    return sents, words
