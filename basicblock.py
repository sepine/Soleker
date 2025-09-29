class BasicBlock:
    def __init__(self, id):
        self.id = id
        self.instructions = []
        self.predecessors = []
        self.successors = []

    def add_instruction(self, instr):
        self.instructions.append(instr)
        instr.block = self

    def __repr__(self):
        instrs = '\n'.join([str(instr) for instr in self.instructions])
        return f"Block {self.id}:\n{instrs}\n"

