import re
from engine.instruction import Instruction


class DisAsmParser:
    def __init__(self, code_str):
        self.code_str = code_str

        self.instructions = []
        self.label_map = {}  

        self.pc = 0
        self.function_raw_map = self.parse_raw_functions()  
        self.pending_label = None 
        self.func_label_map = {}
        self.func2instr = {}

    def parse(self):
        self.pc = 0

        lines = self.code_str.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.endswith(':'):
                label = line.rstrip(':')
                self.pending_label = label
                continue

            instr = self.parse_instruction(line)
            if self.pending_label:
                instr.label = self.pending_label
                self.label_map[self.pending_label] = instr.pc
                self.pending_label = None

            self.instructions.append(instr)
            self.pc += 1

        print("Instructions parsed successfully.")

    def parse_raw_functions(self):
        lines = self.code_str.strip().split('\n')
        current_function = None
        function_map = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.endswith(':') and not line.startswith('lbb_'):
                label = line.rstrip(':')
                current_function = label
                function_map[label] = []
                continue

            function_map[current_function].append(line)

        print("Instructions parsed successfully, with a total of %d functions." % len(function_map))

        return function_map

    def parse_functions(self):
        for func_name, instruction_list in self.function_raw_map.items():
            self.parse_per_function(func_name, instruction_list)

    def parse_per_function(self, func_name, instruction_list):

        self.func_label_map[func_name] = {}
        self.func2instr[func_name] = []

        for line in instruction_list:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.endswith(':'):
                label = line.rstrip(':')
                self.pending_label = label
                continue

            instr = self.parse_instruction(line)
            if self.pending_label:
                instr.label = self.pending_label
                self.func_label_map[func_name][self.pending_label] = instr.pc
                self.pending_label = None

            self.func2instr[func_name].append(instr)
            self.pc += 1

    def parse_instruction(self, line):
        split_parts = line.split('[sep]')
        assert len(split_parts) == 3
        instr_offset, bin_op, instr_str = split_parts[0], split_parts[1], split_parts[2]

        instr_offset = '0x' + instr_offset.lower()

        if len(str(bin_op)) == 1:
            bin_op = '0' + bin_op
        bin_op = '0x' + bin_op.lower()

        parts = instr_str.split(None, 1)

        if len(parts) == 1:
            opcode = parts[0]
            operands_str = ''
        else:
            opcode, operands_str = parts

        dst_reg = src_reg = offset = imm = None
        operands = [op.strip() for op in operands_str.split(',')] if operands_str else []

        if opcode in ['add32', 'sub32', 'mul32', 'div32', 'or32', 'and32', 'lsh32', 'rsh32', 'mod32', 'xor32', 'mov32', 'arsh32',
                      'add64', 'sub64', 'mul64', 'div64', 'or64', 'and64', 'lsh64', 'rsh64', 'mod64', 'xor64', 'mov64', 'arsh64',
                      'uhmul64', 'udiv32', 'udiv64', 'urem32', 'urem64', 'lmul32', 'lmul64', 'shmul64',
                      'sdiv32', 'sdiv64', 'srem32', 'srem64', 'hor64']:
        
            if len(operands) != 2:
                raise ValueError(f"Unmatched number of operands: {line}")
            dst_reg = operands[0].strip()
            
            if self.is_register(operands[1]):
                src_reg = operands[1].strip()
            else:
                imm = int(operands[1], 0)

        elif opcode in ['neg32', 'neg64']:
            
            if len(operands) != 1:
                raise ValueError(f"Unmatched number of operands: {line}")
            dst_reg = operands[0].strip()

        elif opcode in ['lddw']:
            
            if len(operands) != 2:
                raise ValueError(f"Unmatched number of operands: {line}")
            dst_reg = operands[0].strip()
            imm = operands[1].strip()

        elif opcode in ['exit', 'ret']:
            pass 

        elif opcode in ['call', 'syscall']:
            if len(operands) != 1:
                raise ValueError(f"Unmatched number of operands: {line}")
            offset = operands[0].strip()

        elif opcode == 'callx':  # TODO check callx
            if len(operands) != 1:
                raise ValueError(f"Unmatched number of operands: {line}")
            
            offset = operands[0].strip()

        elif opcode == 'ja':
            
            if len(operands) != 1:
                raise ValueError(f"Unmatched number of operands: {line}")
            offset = operands[0].strip()

        elif opcode.startswith('j'):
            
            if len(operands) != 3:
                raise ValueError(f"Unmatched number of operands: {line}")
            dst_reg = operands[0].strip()
            
            if self.is_register(operands[1]):
                src_reg = operands[1].strip()
            else:
                imm = int(operands[1], 0)
            offset = operands[2].strip() 

        elif opcode in ['ldxw', 'ldxh', 'ldxb', 'ldxdw']:
            
            if len(operands) != 2:
                raise ValueError(f"Unmatched number of operands: {line}")

            dst_reg = operands[0].strip()
            
            tmp = operands[1].replace(']', '').replace('[', '').strip()
            src_reg = 'r' + str(self.parse_register(tmp))

            offset = tmp[len(src_reg) + 1:].strip()

        elif opcode in ['stw', 'sth', 'stb', 'stdw']:
            
            if len(operands) != 2:
                raise ValueError(f"Unmatched number of operands: {line}")
            imm = int(operands[1], 0)
            tmp = operands[0].replace(']', '').replace('[', '').strip()
            dst_reg = self.parse_register(tmp)

            offset = tmp[len(dst_reg) + 1:]

        elif opcode in ['stxw', 'stxh', 'stxb', 'stxdw']:
            
            if len(operands) != 2:
                raise ValueError(f"Unmatched number of operands: {line}")
            src_reg = operands[1].strip()
            tmp = operands[0].replace(']', '').replace('[', '').strip()
            dst_reg = 'r' + str(self.parse_register(tmp))

            offset = tmp[len(dst_reg) + 1:]

        else:
            raise ValueError(f"Unknown instruction: {opcode}")

        instr = Instruction(
            pc=self.pc,
            code_offset=instr_offset,
            label=None, 
            bin_opcode=bin_op,
            opcode=opcode,
            dst_reg=dst_reg,
            src_reg=src_reg,
            offset=offset,
            imm=imm
        )
        return instr

    def is_register(self, operand):
        return re.match(r'r\d+', operand) is not None

    def is_jump_target(self, operand):
        
        return True if operand.startswith('lbb_') or operand.startswith('function_') else False

    def parse_register(self, operand):
        match = re.match(r'r(\d+)', operand)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Cannot parse register: {operand}")

    def get_instructions(self):
        
        instr_dict = {}
        pc = 0
        for instr in self.instructions:
            instr_dict[pc] = instr
            pc += 1
        return instr_dict

    def get_label_map(self):
        return self.label_map

    def get_func_label_map(self):
        return self.func_label_map

    def get_function_map(self):
        return self.func2instr
