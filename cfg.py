import networkx as nx
import matplotlib.pyplot as plt
import json
import time
import logging


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class BasicBlock:
    def __init__(self, start_instr):
        self.start_instr = start_instr
        self.block_name = start_instr.pc
        self.instructions = []
        self.successors = []  

    def add_instruction(self, instr):
        self.instructions.append(instr)

    def __repr__(self):
        instrs = '\n'.join([str(instr) for instr in self.instructions])
        return f"BasicBlock(start_pc={self.start_instr.pc}):\n{instrs}\n"


class CFGBuilder:
    def __init__(self, instructions, label_map):
        self.instructions = instructions
        self.label_map = label_map
        self.blocks = []
        self.visited_blocks = set()

    def get_cfg(self):
        return {bb.block_name: bb for bb in self.blocks}

    def build_cfg(self):
        if 'entrypoint' not in self.label_map:
            raise ValueError("The entrypoint is not found.")
        entry_pc = self.label_map['entrypoint']
        logger.debug("Building CFG from entrypoint: %s -- %s", self.instructions[entry_pc].code_offset, entry_pc)

        start_time = time.time()
        self._build_cfg_from_pc(entry_pc)
        end_time = time.time()
        logger.info("CFG built in %s seconds.", end_time - start_time)

    def _build_cfg_from_pc(self, pc):
        if pc in self.visited_blocks:
            return
        self.visited_blocks.add(pc)

        instr = self.instructions.get(pc)
        if instr is None:
            logger.error("Instruction not found at PC: %s", pc)
            return

        block = BasicBlock(instr)
        self.blocks.append(block)
        i = pc

        while i < len(self.instructions):
            instr = self.instructions[i]
            block.add_instruction(instr)
            logger.debug("Adding instruction at PC %s with opcode %s", instr.pc, instr.opcode)

            if instr.opcode in ['exit', 'ja', 'call', 'callx', 'syscall', 
                                'jeq', 'jgt', 'jge', 'jset', 'jne', 'jsgt', 'jsge', 'jlt', 'jle', 'jslt', 'jsle']:

                successors = self.get_successors(instr, i)
                block.successors.extend((succ_pc, edge_label, instr.pc) for succ_pc, edge_label in successors)
                logger.debug("Basic block ending at PC %s with successors: %s", instr.pc, [succ for succ in successors])

                for succ_pc, edge_label in successors:
                    self._build_cfg_from_pc(succ_pc)

                if instr.opcode in ['call', 'callx']:
                    pass
                elif instr.opcode == 'syscall' and instr.offset != 'abort':
                    pass
                else:
                    break
            i = self.get_next_pc(i)

    def get_next_pc(self, current_pc):
        len_pc = len(self.instructions)
        return current_pc + 1 if current_pc + 1 < len_pc else None

    def get_successors(self, instr, pc):

        logger.debug('Processing instruction at PC %s with opcode %s', pc, instr.opcode)

        successors = []
        if instr.opcode == 'exit':

            return successors

        elif instr.opcode == 'syscall' and instr.offset == 'abort':

            return successors

        elif instr.opcode == 'call' and 'panic' in str(instr.offset).lower():
            return successors
            
        elif instr.opcode == 'ja':
            target_pc = self.get_target_pc(instr)
            if target_pc is not None:
                successors.append((target_pc, 'jump'))

        elif instr.opcode.startswith('j'):
            target_pc = self.get_target_pc(instr)
            if target_pc is not None:
                successors.append((target_pc, 'jump'))
            next_pc = self.get_next_pc(pc)
            if next_pc is not None:
                successors.append((next_pc, 'next_block'))

        elif instr.opcode == 'call':
            target_pc = self.get_target_pc(instr)
            if target_pc is not None:
                successors.append((target_pc, 'call'))  

        elif instr.opcode == 'callx':
            next_pc = self.get_next_pc(pc)
            if next_pc is not None:
                successors.append((next_pc, 'callx'))

        elif instr.opcode == 'syscall':
            next_pc = self.get_next_pc(pc)
            if next_pc is not None:
                successors.append((next_pc, 'syscall'))
        else:
            next_pc = self.get_next_pc(pc)
            if next_pc is not None:
                successors.append((next_pc, 'next_block'))

        logger.debug("Successors for instruction at PC %s: %s", pc, successors)

        return successors

    def get_target_pc(self, instr):
        target_pc = self.label_map.get(instr.offset)
        if target_pc is None:
            raise ValueError(f"Label not found: {instr.offset}")
        return target_pc

    def print_cfg(self, file_path='./'):

        print(f'{len(self.blocks)} basic blocks found.')

        cfg_data = {}
        for block in self.blocks:
            cfg_data[block.block_name] = {
                "instructions": [str(instr) for instr in block.instructions],
                "successors": [{"target": succ_pc, "label": edge_label, "source": source_pc} for succ_pc, edge_label, source_pc in block.successors],
            }
        with open(f"{file_path}/cfg.json", "w") as f:
            json.dump(cfg_data, f, indent=2)