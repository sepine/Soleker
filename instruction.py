class Instruction:
    def __init__(self, pc, opcode, dst_reg, src_reg, offset, imm):

        self.pc = pc                    
        self.opcode = opcode            
        self.dst_reg = dst_reg        
        self.src_reg = src_reg          
        self.offset = offset            
        self.imm = imm                  


        self.read_regs = []           
        self.write_regs = []           

        self.block = None               
        self.successors = []          
