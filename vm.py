from sym.instruction import Instruction
from collections import defaultdict
from sym.op import opcode_info
from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx
import json
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class VM:
    def __init__(self, file_dir, max_depth=100):

        self.file_dir = file_dir 
        self.dfg = self._parse_dfg(self.file_dir)
        self.memory = defaultdict(lambda: None)  
        self.node_states = {} 
        self.max_depth = max_depth  
        self.visited = defaultdict(int)  

        self._init_registers()

    def _init_registers(self):
        self.registers = {
            'r0': "RET",
            'r1': "ARG1", 'r2': "ARG2", 'r3': "ARG3", 'r4': "ARG4", 'r5': "ARG5",
            'r6': "CLP6", 'r7': "CLP7", 'r8': "CLP8", 'r9': "CLP9",
            'r10': "FRP",
            'r11': "STP"
        }

    def _parse_instruction(self, node):
        return Instruction(
            pc=node.get("id"),
            opcode=node.get("opcode"),
            dst_reg=node.get("dst"),
            src_reg=node.get("src"),
            offset=node.get("offset"),
            imm=node.get("imm"),
        )

    def _parse_dfg(self, dfg_file):
        with open(dfg_file, 'r') as f:
            dfg = json.load(f)

        self.nodes = {
            node["id"]: self._parse_instruction(node) for node in dfg.get("nodes", [])
        }

        self.edges = dfg.get("links", [])

        return {"nodes": self.nodes, "edges": self.edges}

    def execute(self):
        if not self.dfg['nodes']:
            raise RuntimeError("The DFG does not contain any nodes to execute.")

        keys = list(self.dfg['nodes'].keys())
        start_node = self.dfg['nodes'][keys[0]] 
        self._execute_node(start_node, current_depth=0)

    def _execute_node(self, node, current_depth):
        node_id = node.pc

        if current_depth > self.max_depth:
            return

        if self.visited[node_id] >= 2:
            return

        self.visited[node_id] += 1

        instruction = node

        if node_id not in self.node_states:
            self.node_states[node_id] = {
                'memory': {},
                'exec': []
            }

        self.execute_instruction(instruction)
        self.node_states[node_id]['memory'] = dict(self.memory)

        for edge in self.dfg['edges']:
            if edge['source'] == node_id:
                next_node_id = edge['target']
                next_node = self.dfg['nodes'][next_node_id]
                saved_memory = dict(self.memory)
                self._execute_node(next_node, current_depth + 1)
                self.memory = saved_memory

    def execute_instruction(self, instr):
        is_imm = False
        if instr.imm != "[NULL]":
            is_imm = True

        op_name = instr.opcode

        if op_name in ['call', 'callx', 'syscall']:  
            self.deal_call(instr, op_name)
        elif op_name == 'exit':
            self.deal_exit(instr, op_name)
        elif op_name.startswith('ldx'): 
            self.deal_load(instr, op_name)
        elif op_name.startswith('st'):  
            self.deal_store(instr, op_name)
        elif op_name.startswith('j'):
            self.deal_jump(instr, op_name)
        elif op_name.startswith('neg'):
            self.deal_unary(instr, op_name)
        elif is_imm:
            self.deal_imm(instr, op_name)
        elif is_imm == False:
            self.deal_reg(instr, op_name)
        else:
            raise ValueError(f"Unhandled instruction: {op_name}")

    def deal_unary(self, instr, op_name: str):
        dst = instr.dst_reg
        instr_pc = instr.pc
        dst_val = self.registers.get(dst)

        if 'neg' in op_name:
            result = f"{dst_val} NEG"
        else:
            raise ValueError(f"Unsupported operation: {op_name}")

        self.registers[dst] = result

        self.node_states[instr_pc]['exec'].append(f"{op_name} {result}")

    def deal_imm(self, instr, op_name: str):
        dst = instr.dst_reg
        instr_pc = instr.pc

        imm = instr.imm
        if imm.startswith('0x'):
            pass
        else:
            imm = int(imm)
        if op_name.startswith('mov'):
            self.registers[dst] = imm
            self.node_states[instr_pc]['exec'].append(f"{imm} {op_name} {imm}")
        elif op_name.startswith(('add', 'sub', 'mul', 'div', 'or', 'and', 'lsh', 'rsh', 'mod', 'xor', 'arsh')):
            dst_val = self.registers.get(dst)
            if 'add' in op_name:
                result = f"{dst_val} + {imm}" if imm >= 0 else f"{dst_val} - {-imm}"
            elif 'sub' in op_name:
                result = f"{dst_val} - {imm}" if imm >= 0 else f"{dst_val} + {-imm}"
            elif 'mul' in op_name:
                result = f"{dst_val} * {imm}" if imm >= 0 else f"{dst_val} * ({imm})"
            elif 'div' in op_name:
                result = f"{dst_val} // {imm}" if imm >= 0 else f"{dst_val} // ({imm})"
            elif 'mod' in op_name:
                result = f"{dst_val} % {imm}" if imm >= 0 else f"{dst_val} % ({imm})"
            elif 'or' in op_name:
                result = f"{dst_val} | {imm}" if imm >= 0 else f"{dst_val} | ({imm})"
            elif 'and' in op_name:
                result = f"{dst_val} & {imm}" if imm >= 0 else f"{dst_val} & ({imm})"
            elif 'xor' in op_name:
                result = f"{dst_val} ^ {imm}" if imm >= 0 else f"{dst_val} ^ ({imm})"
            elif 'lsh' in op_name:
                result = f"{dst_val} << {imm}" if imm >= 0 else f"{dst_val} << ({imm})"
            elif 'rsh' in op_name:
                result = f"{dst_val} >> {imm}" if imm >= 0 else f"{dst_val} >> ({imm})"
            elif 'arsh' in op_name: 
                result = f"{dst_val} [>>] {imm}" if imm >= 0 else f"{dst_val} [>>] ({imm})"
            else:
                raise ValueError(f"Unsupported operation: {op_name}")
            self.registers[dst] = result
            self.node_states[instr_pc]['exec'].append(f"{result} {op_name} {dst_val} {imm}")
        elif op_name in ['uhmul64','udiv32', 'udiv64', 'urem32', 'urem64', 'lmul32', 'lmul64',
                         'shmul64', 'sdiv32', 'sdiv64', 'srem32', 'srem64']:
            dst_val = self.registers.get(dst)
            if op_name == 'uhmul64':  
                result = f"{dst_val} * {imm} >>64" if imm >= 0 else f"{dst_val} * ({imm}) >>64"
            elif op_name in ['lmul32', 'lmul64']: 
                result = f"{dst_val} * {imm}" if imm >= 0 else f"{dst_val} * ({imm})"
            elif op_name == 'shmul64':
                result = f"{dst_val} * {imm} >>64&F" if imm >= 0 else f"{dst_val} * ({imm}) >>64&F"
            elif op_name in ['udiv32', 'udiv64']: 
                result = f"{dst_val} // {imm}" if imm >= 0 else f"{dst_val} // ({imm})"
            elif op_name in ['sdiv32', 'sdiv64']:
                result = f"{dst_val} // {imm}" if imm >= 0 else f"{dst_val} // ({imm})"
            elif op_name in ['urem32', 'urem64']:
                result = f"{dst_val} % {imm}" if imm >= 0 else f"{dst_val} % ({imm})"
            elif op_name in ['srem32', 'srem64']:  
                result = f"{dst_val} % {imm}" if imm >= 0 else f"{dst_val} % ({imm})"
            else:
                raise ValueError(f"Unsupported operation: {op_name}")
            self.registers[dst] = result
            self.node_states[instr_pc]['exec'].append(f"{result} {op_name} {dst_val} {imm}")
        elif op_name == 'lddw':
            imm_val = f"[LD_IMM]{str(instr.imm)}"
            self.registers[dst] = imm_val
            self.node_states[instr_pc]['exec'].append(f"{imm_val} {op_name} {str(instr.imm)}")
        elif op_name in ['le', 'hor64']:
            dst_val = self.registers.get(dst)
            if op_name == 'le':
                result = f"{dst_val} &F" 
                self.registers[dst] = result
                self.node_states[instr_pc]['exec'].append(f"{result} {op_name} {dst_val}")
            elif op_name == 'hor64':
                imm_val = instr.imm  
                high_bits = f"{imm_val} &F<<32" if imm_val >= 0 else f"({imm_val}) &F<<32"
                result = f"{dst_val} | {high_bits}"
                self.registers[dst] = result
                self.node_states[instr_pc]['exec'].append(f"{result} {op_name} {str(imm_val)}")
        else:
            raise ValueError(f"Unhandled immediate operation: {op_name}")
    
    def read_memory(self, address):
        if isinstance(address, str): 
            if address in self.memory:
                return self.memory[address]
            else:
                return f"UNKNOWN_{address}"  
        elif isinstance(address, int):  
            addr_str = hex(address)
            if addr_str in self.memory:
                return self.memory[addr_str]
            else:
                return f"UNINITIALIZED_{addr_str}" 
        else:
            raise ValueError(f"Invalid address type: {address}")

    def write_memory(self, address, value):
        if isinstance(address, str): 
            self.memory[address] = value
        elif isinstance(address, int):  
            addr_str = hex(address)
            self.memory[addr_str] = value
        else:
            raise ValueError(f"Invalid address type: {address}")
        
    def deal_reg(self, instr, op_name: str):
        instr_pc = instr.pc
        dst = instr.dst_reg
        src = instr.src_reg
        if op_name.startswith('mov'):
            src_val = self.registers.get(src)
            self.registers[dst] = src_val
            self.node_states[instr_pc]['exec'].append(f"{src_val} {op_name} {src_val}")
        elif op_name.startswith(('add', 'sub', 'mul', 'div', 'or', 'and', 'lsh', 'rsh', 'mod', 'xor', 'arsh')):
            src_val = self.registers.get(src)
            dst_val = self.registers.get(dst)
            if 'add' in op_name:
                result = f"{dst_val} + {src_val}"
            elif 'sub' in op_name:
                result = f"{dst_val} - {src_val}"
            elif 'mul' in op_name:
                result = f"{dst_val} * {src_val}"
            elif 'div' in op_name:
                result = f"{dst_val} // {src_val}"
            elif 'mod' in op_name:
                result = f"{dst_val} % {src_val}"
            elif 'or' in op_name:
                result = f"{dst_val} | {src_val}"
            elif 'and' in op_name:
                result = f"{dst_val} & {src_val}"
            elif 'xor' in op_name:
                result = f"{dst_val} ^ {src_val}"
            elif 'lsh' in op_name:
                result = f"{dst_val} << {src_val}"
            elif 'rsh' in op_name:
                result = f"{dst_val} >> {src_val}"
            elif 'arsh' in op_name:
                result = f"{dst_val} [>>] {src_val}" 
            else:
                raise ValueError(f"Unsupported operation: {op_name}")
            self.registers[dst] = result
            self.node_states[instr_pc]['exec'].append(f"{result} {op_name} {dst_val} {src_val}")

        elif op_name in ['uhmul64', 'udiv32',  'udiv64', 'urem32', 'urem64', 'lmul32', 'lmul64',
                         'shmul64', 'sdiv32', 'sdiv64', 'srem32', 'srem64']:
            src_val = self.registers.get(src)
            dst_val = self.registers.get(dst)
            if op_name == 'uhmul64': 
                result = f"{dst_val} * {src_val} >>64"
            elif op_name in ['lmul32', 'lmul64']: 
                result = f"{dst_val} * {src_val}"
            elif op_name == 'shmul64':
                result = f"{dst_val} * {src_val} >>64&F"
            elif op_name in ['udiv32', 'udiv64']: 
                result = f"{dst_val} // {src_val}"
            elif op_name in ['sdiv32', 'sdiv64']:  
                result = f"{dst_val} // {src_val}"
            elif op_name in ['urem32', 'urem64']:  
                result = f"{dst_val} % {src_val}"
            elif op_name in ['srem32', 'srem64']: 
                result = f"{dst_val} % {src_val}"
            else:
                raise ValueError(f"Unsupported operation: {op_name}")
            self.registers[dst] = result
            self.node_states[instr_pc]['exec'].append(f"{result} {op_name} {dst_val} {src_val}")
        elif op_name == 'lddw':
            raise NotImplementedError(f"Unhandled lddw operation for register ")
        elif op_name == 'be':
            dst_val = self.registers.get(dst)  
            result = f"{dst_val}.swap_bytes()"
            self.registers[dst] = result
            self.node_states[instr_pc]['exec'].append(f"{result} {op_name} {dst_val}")
        else:
            raise ValueError(f"Unhandled register operation: {op_name}")

    def deal_load(self, instr, op_name: str):
        dst = instr.dst_reg
        src = instr.src_reg
        instr_pc = instr.pc
        offset = instr.offset
        address = self.registers[src]
        src_address = f"mem[{address}_{offset}]"  
        value = self.read_memory(src_address)  
        self.registers[dst] = value  
        self.node_states[instr_pc]['exec'].append(f"{value} {op_name} {src_address}")

    def deal_store(self, instr, op_name: str):
        src = instr.src_reg
        dst = instr.dst_reg
        instr_pc = instr.pc
        offset = instr.offset
        instr.read_regs = [src, dst]
        address = self.registers[dst]
        dst_address = f"mem[{address}_{offset}]"  
        value = self.registers[src] 
        self.write_memory(dst_address, value) 
        self.node_states[instr_pc]['exec'].append(f"{op_name} {dst_address} {value}")

    def deal_call(self, instr, op_name: str):
        instr_id = instr.pc  
        offset = instr.offset 
        if op_name == 'call': 
            target = "INTERNAL_FUNCTION"
            self.node_states[instr_id]['exec'].append(f"{op_name} {target}")
        elif op_name == 'callx':
            target = self.registers[offset]
            self.node_states[instr_id]['exec'].append(f"{op_name} {target}")
        elif op_name == 'syscall':
            self.node_states[instr_id]['exec'].append(f"{op_name} {offset}")
        else:
            raise ValueError(f"Unhandled call operation: {op_name}")

    def deal_jump(self, instr, op_name: str):
        instr_pc = instr.pc
        instr.read_regs = []
        instr.write_regs = []
        if op_name in ['jeq', 'jgt', 'jge', 'jset', 'jne', 'jsgt', 'jsge', 'jlt', 'jle', 'jslt', 'jsle']:
            dst = instr.dst_reg
            src = instr.src_reg  
            dst_val = self.registers.get(dst)
            if src is not None and src.startswith('r'):
                src_val = self.registers.get(src)
                result = f"{op_name} {dst_val} {src_val}"
            else:
                imm = instr.imm
                result = f"{op_name} {dst_val} {imm}"

            self.node_states[instr_pc]['exec'].append(result)
        else:
            result = f"{op_name}"
            self.node_states[instr_pc]['exec'].append(result)

    def deal_exit(self, instr, op_name: str):
        instr_pc = instr.pc
        self.node_states[instr_pc]['exec'].append(f"{op_name}")
