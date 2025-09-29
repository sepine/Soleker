import json


def escape_string(s):
    """
    Escapes double quotes and backslashes in strings for Souffle.
    """
    if s is None:
        return "[NULL]"
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def generate_souffle_facts(dfg, nodes_fact_file='souffle/facts/node.facts',
                           edges_fact_file='souffle/facts/edge.facts'):
    with open(nodes_fact_file, 'w') as nf:
        index = 0
        for node, attrs in dfg.nodes(data=True):
 
            pc = attrs.get('pc')
            opcode = escape_string(attrs.get('opcode'))

            operands = attrs.get('operands', {})
            dst = escape_string(operands.get('dst')) 
            src = escape_string(operands.get('src'))
            offset = operands.get('offset') 
            imm = operands.get('imm') 

            offset_str = "[NULL]" if offset is None else str(offset)

            imm_str = escape_string(str(imm)) if imm is not None else "[NULL]"
 
            fact = f'{index}\t{pc}\t{opcode}\t{dst}\t{src}\t{offset_str}\t{imm_str}\n'
            nf.write(fact)
            index += 1

    print(f"Node facts written to {nodes_fact_file}")

    with open(edges_fact_file, 'w') as ef:

        for src_pc, dst_pc, attrs in dfg.edges(data=True):

            edge_type = escape_string(attrs.get('label'))
            fact = f'{src_pc}\t{dst_pc}\t{edge_type}\n'
            ef.write(fact)

    print(f"Edge facts written to {edges_fact_file}")
