import os, re

import pexpect

from wsd_config import concraft_model_path

def parse_morfeusz_output(morfeusz_str):
    # replace unwieldy special characters to make parsing reliable
    ###morfeusz_str = re.sub('([\\[,])\\[(?=,)', '\\1MORFEUSZ_LEFT_SQUARE_BRACKET', morfeusz_str)
    ###morfeusz_str = re.sub('([\\[,])\\](?=,)', '\\1MORFEUSZ_RIGHT_SQUARE_BRACKET', morfeusz_str)
    morfeusz_str = re.sub(',,', ',MORFEUSZ_COMMA', morfeusz_str)
    morfeusz_str = re.sub('\\.,', 'MORFEUSZ_DOT,', morfeusz_str)

    sections = morfeusz_str.split(']\n[')
    sections[0] = sections[0].strip()[1:] # remove the first and last bracket
    sections[-1] = sections[-1].strip()[:-1]

    parsed_sections = []
    for sec in sections:
        nodes = [node.split(',') for node in sec.split('\n')]
        parsed_nodes = []
        for (node_n, items) in enumerate(nodes): # 'natively-printed' Morfeusz options for this graph path
            clean_items = [re.sub('^_$', '', item.strip().strip('[]').replace('MORFEUSZ_COMMA', ','))
                                    for item in items]
            clean_items[2] = clean_items[2].replace('MORFEUSZ_DOT', '.') # form and lemma
            clean_items[3] = clean_items[3].replace('MORFEUSZ_DOT', '.')
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

def write_dag_from_morfeusz(path, morfeusz_nodes):
    with open(path, 'w+') as out:
        for (node_n, node) in enumerate(morfeusz_nodes):
            for variant in node:
                if node_n < (len(morfeusz_nodes) - 1):
                    concraft_columns = [str(1/len(node)), '', '']
                else: # add end of sentence tag
                    concraft_columns = [str(1/len(node)), '', 'eos']
                print('\t'.join(variant + concraft_columns), file=out)
        print('', file=out)

def parse_concraft_output(path):
    concraft = pexpect.spawn('concraft-pl tag -i {} {}'.format(path, concraft_model_path))
    pexp_result = concraft.expect(['\r\n\r\n', pexpect.EOF, pexpect.TIMEOUT])
    if pexp_result != 0:
        raise RuntimeError('there was a Concraft error: {}'.format(concraft.before))
    concraft_interp = concraft.before.decode()
    token_tuples = []
    for line in concraft_interp.split('\n'):
        if re.search('\t1.0000\t', line):
            fields = line.split('\t')
            token_tuples.append(tuple(fields[2:5]))
    return token_tuples

morfeusz_analyzer = pexpect.spawn('morfeusz_analyzer')
pexp_result = morfeusz_analyzer.expect(['Using dictionary: [^\\n]*$', pexpect.EOF, pexpect.TIMEOUT])
if pexp_result != 0:
    raise RuntimeError('cannot run morfeusz_analyzer properly')

def parse_sentence(sent):
    """Use Morfeusz and Concraft to obtain the sentence as a list of (form, lemma, interp)"""
    morfeusz_analyzer.send(sent+' KONIECKONIEC\n')
    pexp_result = morfeusz_analyzer.expect(['\r\n\\[\\d+,\\d+,KONIECKONIEC,KONIECKONIEC,ign,_,_\\]\r\n', pexpect.EOF, pexpect.TIMEOUT])
    if pexp_result != 0:
        raise RuntimeError('there was a Morfeusz error: {}'.format(morfeusz_analyzer.before))
    morfeusz_interp = morfeusz_analyzer.before.decode().strip() # encode from bytes into str, strip whitespace
    morfeusz_interp = morfeusz_interp[morfeusz_interp.index('['):]
    parsed_nodes = parse_morfeusz_output(morfeusz_interp)
    write_dag_from_morfeusz('MORFEUSZ_CONCRAFT_TEMP', parsed_nodes)
    parsed_sent = parse_concraft_output('MORFEUSZ_CONCRAFT_TEMP')
    os.remove('MORFEUSZ_CONCRAFT_TEMP')
    return parsed_sent
