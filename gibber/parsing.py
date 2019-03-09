import os, re

import pexpect

from wsd_config import concraft_model_path

def parse_morfeusz_output(morfeusz_str):
    # replace unwieldy special characters to make parsing reliable
    morfeusz_str = re.sub(',,', ',MORFEUSZ_COMMA', morfeusz_str)
    morfeusz_str = re.sub('\\.,', 'MORFEUSZ_DOT,', morfeusz_str)

    sections = morfeusz_str.split(']\r\n[') # pexpect uses \r\n style newlines
    sections[0] = sections[0].strip()[1:] # remove the first and last bracket
    sections[-1] = sections[-1].strip()[:-1]

    parsed_sections = []
    for sec in sections: # (per graph path, which can have some alternatives inside)
        nodes = [node.split(',') for node in sec.split('\n')]
        parsed_nodes = []
        for (node_n, items) in enumerate(nodes): # 'natively-printed' Morfeusz options for this graph path
            clean_items = [re.sub('^_$', '', item.strip().strip('[]').replace('MORFEUSZ_COMMA', ','))
                                    for item in items]
            clean_items[2] = clean_items[2].replace('MORFEUSZ_DOT', '.') # form and lemma
            clean_items[3] = clean_items[3].replace('MORFEUSZ_DOT', '.')
            if len(clean_items) > 7: # some wild commas, just fake the node with something inoffensive
                clean_items = clean_items[0:2] + ['_', '_', 'ign', '', '']
            assembled_alternatives = []
            for (pos_n, alts) in enumerate(clean_items[4].split(':')):
                new_alternatives = []
                alts = alts.split('.')
                for alt in alts:
                    if pos_n == 0:
                        new_alternatives.append(alt)
                    else:
                        for prev_alt in assembled_alternatives:
                            new_alternatives.append(prev_alt+':'+alt)
                assembled_alternatives = new_alternatives
            for tag in assembled_alternatives:
                parsed_nodes.append(clean_items[:4] + [ tag ] + clean_items[5:])
        parsed_sections.append(parsed_nodes)
    return parsed_sections

def split_morfeusz_sents(morfeusz_nodes, verbose=False):
    """Given an output from parse_morfeusz_output, return it as a list of sentences."""
    sent_boundaries = [0]
    previous_brev = False
    for (node_n, node) in enumerate(morfeusz_nodes):
        current_brev = False
        for variant in node:
            if 'brev' in variant[4]:
                previous_brev = True
                current_brev = True
            if variant[2] == '.' and not previous_brev:
                sent_boundaries.append(node_n+1)
        if not current_brev:
            previous_brev = False
    sent_boundaries.append(len(morfeusz_nodes))
    if verbose:
        print('sentence boundaries', sent_boundaries)
    sents = []
    for (bnd_n, bnd) in enumerate(sent_boundaries[1:]):
        sents.append(morfeusz_nodes[sent_boundaries[bnd_n]:bnd]) # bnd_n is effectively bnd_n-1, because of skipping first element
    sents = [s for s in sents if len(s) > 0]
    return sents

def write_dag_from_morfeusz(path, morfeusz_nodes, append_sentence=False):
    open_settings = 'a' if append_sentence else 'w+'
    with open(path, open_settings) as out:
        for (node_n, node) in enumerate(morfeusz_nodes):
            for variant in node:
                if node_n < (len(morfeusz_nodes) - 1):
                    concraft_columns = [str(1/len(node)), '', '']
                else: # add end of sentence tag
                    concraft_columns = [str(1/len(node)), '', 'eos']
                print('\t'.join(variant + concraft_columns), file=out)
        print('', file=out) # newline

def parse_with_concraft(path):
    concraft = pexpect.spawn('concraft-pl tag -i {} {}'.format(path, concraft_model_path))
    pexp_result = concraft.expect([pexpect.EOF, pexpect.TIMEOUT])
    if pexp_result != 0:
        raise RuntimeError('there was a Concraft timeout: {}'.format(concraft.before))
    concraft_interp = concraft.before.decode()
    if 'concraft-pl:' in concraft_interp:
        raise RuntimeError('there was a Concraft error: {}'.format(concraft_interp))
    sents = []
    token_tuples = [] # of the current sentence
    for line in concraft_interp.split('\n'):
        line = line.strip()
        if re.search('\t1.0000\t', line):
            fields = line.split('\t')
            if len(fields) != 11:
                raise RuntimeError('Incorrect number of columns in Concraft output - not 11 -: {}'.format(line))
            token_tuples.append(tuple(fields[2:5]))
        if len(line.strip()) == 0: # end of sentence
            sents.append(token_tuples)
            token_tuples = []
    sents.remove([]) # the last empty "sentence" if present
    return sents

morfeusz_analyzer = pexpect.spawn('morfeusz_analyzer')
pexp_result = morfeusz_analyzer.expect(['Using dictionary: [^\\n]*$', pexpect.EOF, pexpect.TIMEOUT])
if pexp_result != 0:
    raise RuntimeError('cannot run morfeusz_analyzer properly')

def parse_sentences(sents_str, verbose=False):
    """Use Morfeusz and Concraft to obtain the sentences as lists of (form, lemma, interp)"""
    if sents_str.strip() == '':
        raise ValueError('called parse_sentences on empty string')
    sents_str = sents_str.replace('[', '(').replace(']', ')') # square brackets can mess up parse detection in output
    morfeusz_analyzer.send(sents_str+' KONIECKONIEC\n')
    pexp_result = morfeusz_analyzer.expect(['\r\n\\[\\d+,\\d+,KONIECKONIEC,KONIECKONIEC,ign,_,_\\]\r\n', pexpect.EOF, pexpect.TIMEOUT])
    if pexp_result != 0:
        raise RuntimeError('there was a Morfeusz error: {}'.format(morfeusz_analyzer.before))
    morfeusz_interp = morfeusz_analyzer.before.decode().strip() # encode from bytes into str, strip whitespace
    morfeusz_interp = morfeusz_interp[morfeusz_interp.index('['):]
    if verbose:
        print('Morfeusz interp is', morfeusz_interp)
    parsed_nodes = parse_morfeusz_output(morfeusz_interp)
    if verbose:
        print(len(parsed_nodes), 'parsed nodes')

    morfeusz_sentences = split_morfeusz_sents(parsed_nodes, verbose=verbose)
    if verbose:
        print('Morfeusz sentences,', len(morfeusz_sentences), ':', morfeusz_sentences)
    for sent_n, morf_sent in enumerate(morfeusz_sentences):
        if sent_n == 0:
            write_dag_from_morfeusz('MORFEUSZ_CONCRAFT_TEMP', morf_sent)
        else:
            write_dag_from_morfeusz('MORFEUSZ_CONCRAFT_TEMP', morf_sent, append_sentence=True)
    parsed_sents = parse_with_concraft('MORFEUSZ_CONCRAFT_TEMP')
    os.remove('MORFEUSZ_CONCRAFT_TEMP')
    return parsed_sents

def extract_samples(string):
    """Get the fragments of string that constitute language samples containing the unit in
    question. Works for Polish Wordnet 3."""
    definition = re.search('##D: (.*?) [\[#]', string)
    if definition is not None and re.search('\\w', definition.group(1)): # ensure that we catch something alphabetic, not only special chars
        definition = definition.group(1)
    else:
        definition = ''
    example = re.search('##W: (.*?) [\[#]', string)
    if example is not None and re.search('\\w', example.group(1)):
        example = example.group(1)
    else:
        example = ''
    return definition + ' ' + example
