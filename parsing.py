import re

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
            clean_items = [re.sub('^_$', '', item.strip().replace('MORFEUSZ_COMMA', ','))
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
                if node_n < (len(nodes) - 1):
                    concraft_columns = [1/len(nodes), '', '']
                else: # add end of sentence tag
                    concraft_columns = [1/len(nodes), '', 'eos']
                print('\t'.join(variant + concraft_columns), file=out)
        print('', file=out)
