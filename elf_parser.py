from elftools.elf.elffile import ELFFile
from elftools.elf.sections import (
    SymbolTableSection,
    StringTableSection,
    Section,
    NullSection,
    NoteSection,
    Symbol
)
from elftools.elf.relocation import RelocationSection
from elftools.elf.dynamic import DynamicSection

class ELFParser:
    def __init__(self, filename):
        self.filename = filename
        self.sections = {}
        self.dwarf_info = None  

    def parse(self):
        with open(self.filename, 'rb') as f:
            elffile = ELFFile(f)

            if elffile.has_dwarf_info():
                self.dwarf_info = elffile.get_dwarf_info()

            for section in elffile.iter_sections():
                name = section.name
                if name == '.text':
                    continue  

                if isinstance(section, SymbolTableSection):
                    symbols = []
                    for symbol in section.iter_symbols():
                        symbols.append({
                            'name': symbol.name,
                            'info': symbol['st_info']['type'],
                            'bind': symbol['st_info']['bind'],
                            'shndx': symbol['st_shndx'],
                            'value': symbol['st_value'],
                            'size': symbol['st_size'],
                        })
                    self.sections[name] = symbols
                elif isinstance(section, StringTableSection):

                    data = section.data()
                    strings = data.decode('utf-8', errors='replace').split('\x00')
                    self.sections[name] = strings
                elif isinstance(section, RelocationSection):
         
                    relocations = []
                    sym_table = None
                    if section['sh_link']:
                        sym_table = elffile.get_section(section['sh_link'])
                    for relocation in section.iter_relocations():
                        reloc_info = relocation['r_info']
                        symbol = relocation['r_info_sym']
                        symbol_name = ''
                        if sym_table:
                            symbol_entry = sym_table.get_symbol(symbol)
                            symbol_name = symbol_entry.name
                        relocations.append({
                            'offset': relocation['r_offset'],
                            'info': reloc_info,
                            'type': relocation['r_info_type'],
                            'symbol_index': symbol,
                            'symbol_name': symbol_name,
                        })
                    self.sections[name] = relocations
                elif isinstance(section, DynamicSection):
                 
                    dynamics = []
                    for tag in section.iter_tags():
                        dynamics.append({
                            'tag': tag.entry.d_tag,
                            'value': tag.entry.d_val,
                            'info': tag.entry
                        })
                    self.sections[name] = dynamics
                elif name == '.gnu.hash':
                   
                    gnu_hash = self.parse_gnu_hash(section)
                    self.sections[name] = gnu_hash
                elif name == '.hash':
                   
                    sysv_hash = self.parse_sysv_hash(section)
                    self.sections[name] = sysv_hash
                elif name == '.shstrtab':
                 
                    data = section.data()
                    strings = data.decode('utf-8', errors='replace').split('\x00')
                    self.sections[name] = strings
                elif name in ['.rodata', '.data.rel.ro', '.data']:
                    
                    data = section.data()

                    strings = data.decode('ascii', errors='replace').split('\x00')
                    
                    strings = [s for s in strings if s]
                    self.sections[name] = {
                        'raw_data': data,
                        'strings': strings
                    }
                elif name == '.eh_frame':
                 
                    eh_frame = []
                    
                    eh_frame_data = section.data()
                    self.sections[name] = eh_frame_data
                elif isinstance(section, NullSection) and section.header['sh_name'] == 0:
                    
                    self.sections['NULL'] = "Empty section (Index 0)"
                else:
                   
                    data = section.data()
                    self.sections[name] = data

    def parse_gnu_hash(self, section):
        data = section.data()
      
        nbuckets = int.from_bytes(data[0:4], 'little')
        symoffset = int.from_bytes(data[4:8], 'little')
        bloom_size = int.from_bytes(data[8:12], 'little')
        bloom_shift = int.from_bytes(data[12:16], 'little')

   
        gnu_hash_info = {
            'nbuckets': nbuckets,
            'symoffset': symoffset,
            'bloom_size': bloom_size,
            'bloom_shift': bloom_shift,

        }
        return gnu_hash_info

    def parse_sysv_hash(self, section):
        data = section.data()
 
        nbuckets = int.from_bytes(data[0:4], 'little')
        nchains = int.from_bytes(data[4:8], 'little')

        sysv_hash_info = {
            'nbuckets': nbuckets,
            'nchains': nchains,

        }
        return sysv_hash_info

    def get_sections(self):
        return self.sections
