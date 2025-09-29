# Mapping of opcodes to mnemonics

group_imm = {
    0x00: 'lddw',

    0x61: 'ldxw',
    0x71: 'ldxb',

    0x62: 'stw',
    0x72: 'stb',

    0x63: 'stxw',
    0x73: 'stxb',

    0x04: 'add32',
    0x14: 'sub32',
    0x24: 'mul32',
    0x34: 'div32',
    0x44: 'or32',
    0x54: 'and32',
    0x64: 'lsh32',
    0x74: 'rsh32',
    0x84: 'neg32',
    0x94: 'mod32',
    0xa4: 'xor32',
    0xb4: 'mov32',
    0xc4: 'arsh32',
    0xd4: 'le',

    0x05: 'ja',
    0x15: 'jeq',
    0x25: 'jgt',
    0x35: 'jge',
    0x45: 'jset',
    0x55: 'jne',
    0x65: 'jsgt',
    0x75: 'jsge',
    0x85: 'call',  
    0x95: 'exit',
    0xa5: 'jlt',
    0xb5: 'jle',
    0xc5: 'jslt',
    0xd5: 'jsle',

    0x36: 'uhmul64',
    0x46: 'udiv32',
    0x56: 'udiv64',
    0x66: 'urem32',
    0x76: 'urem64',
    0x86: 'lmul32',
    0x96: 'lmul64',
    0xb6: 'shmul64',
    0xc6: 'sdiv32',
    0xd6: 'sdiv64',
    0xe6: 'srem32',
    0xf6: 'srem64',

    0x07: 'add64',
    0x17: 'sub64',
    0x27: 'mul64',
    0x37: 'div64',
    0x47: 'or64',
    0x57: 'and64',
    0x67: 'lsh64',
    0x77: 'rsh64',
    0x87: 'neg64',
    0x97: 'mod64',
    0xa7: 'xor64',
    0xb7: 'mov64',
    0xc7: 'arsh64',
    0xf7: 'hor64',
}

group_reg = {
    0x18: 'lddw',

    0x69: 'ldxh',
    0x79: 'ldxdw',

    0x6a: 'sth',
    0x7a: 'stdw',

    0x6b: 'stxh',
    0x7b: 'stxdw',

    0x0c: 'add32',
    0x1c: 'sub32',
    0x2c: 'mul32',
    0x3c: 'div32',
    0x4c: 'or32',
    0x5c: 'and32',
    0x6c: 'lsh32',
    0x7c: 'rsh32',

    0x9c: 'mod32',
    0xac: 'xor32',
    0xbc: 'mov32',
    0xcc: 'arsh32',
    0xdc: 'be',

    0x1d: 'jeq',
    0x2d: 'jgt',
    0x3d: 'jge',
    0x4d: 'jset',
    0x5d: 'jne',
    0x6d: 'jsgt',
    0x7d: 'jsge',
    0x8d: 'callx',

    0xad: 'jlt',
    0xbd: 'jle',
    0xcd: 'jslt',
    0xdd: 'jsle',

    0x3e: 'uhmul64',
    0x4e: 'udiv32',
    0x5e: 'udiv64',
    0x6e: 'urem32',
    0x7e: 'urem64',
    0x8e: 'lmul32',
    0x9e: 'lmul64',

    0xbe: 'shmul64',
    0xce: 'sdiv32',
    0xde: 'sdiv64',
    0xee: 'srem32',
    0xfe: 'srem64',

    0x0f: 'add64',
    0x1f: 'sub64',
    0x2f: 'mul64',
    0x3f: 'div64',
    0x4f: 'or64',
    0x5f: 'and64',
    0x6f: 'lsh64',
    0x7f: 'rsh64',

    0x9f: 'mod64',
    0xaf: 'xor64',
    0xbf: 'mov64',
    0xcf: 'arsh64',
}

groups = {
    'group_imm': group_imm,
    'group_reg': group_reg,
}


def format_hex(number):
    return f"0x{number:02x}"

opcode_info = {}
for opcode_num, op_name in group_imm.items():
    opcode_info[format_hex(opcode_num)] = {'name': op_name, 'group': 'imm'}

for opcode_num, op_name in group_reg.items():
    opcode_info[format_hex(opcode_num)] = {'name': op_name, 'group': 'reg'}
