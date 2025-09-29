import networkx as nx
import pandas as pd
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt


def is_csv_empty(file_path):
    """Check if a CSV file is empty or contains only headers."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df.empty
    except pd.errors.EmptyDataError:
        return True
    except Exception as e:
        return True
    

def read_syscall_results(results_dir):
    syscall_pairs = {}

    for filename in os.listdir(results_dir):
        if not filename.endswith('.csv'):
            continue
        
        file_path = os.path.join(results_dir, filename)
        if is_csv_empty(file_path):
            continue

        name = filename[:-4]  
        if '_influences' in name:

            syscall_name = name.split('_influences')[0]
            if syscall_name not in syscall_pairs:
                syscall_pairs[syscall_name] = {'forward': None, 'backward': None}
            syscall_pairs[syscall_name]['forward'] = filename
        elif 'influences_' in name:
            syscall_name = name.split('influences_')[1]
            if syscall_name not in syscall_pairs:
                syscall_pairs[syscall_name] = {'forward': None, 'backward': None}
            syscall_pairs[syscall_name]['backward'] = filename
    
    valid_syscalls = {}
    for syscall_name, files in syscall_pairs.items():
        if files['forward'] is not None or files['backward'] is not None:
            valid_syscalls[syscall_name] = files
    
    return valid_syscalls


def get_influence_nodes(results_dir, forward_file, backward_file):
    forward_nodes = set()
    backward_nodes = set()
    
    if forward_file:
        forward_df = pd.read_csv(os.path.join(results_dir, forward_file), sep='\t')
        for _, row in forward_df.iterrows():
            forward_nodes.update(row)
    
    if backward_file:
        backward_df = pd.read_csv(os.path.join(results_dir, backward_file), sep='\t')
        for _, row in backward_df.iterrows():
            backward_nodes.update(row)
    
    return forward_nodes, backward_nodes


def build_syscall_graph(facts_dir, results_dir, syscall_name, influence_files):
    G = nx.DiGraph()
    nodes_df = pd.read_csv(os.path.join(facts_dir, 'node.facts'), 
                          sep='\t', 
                          names=['index', 'pc', 'opcode', 'dst', 'src', 'offset', 'imm'])
    
    edges_df = pd.read_csv(os.path.join(facts_dir, 'edge.facts'), 
                          sep='\t', 
                          names=['src_pc', 'dst_pc', 'edge_type'])
    
    forward_nodes, backward_nodes = get_influence_nodes(results_dir, 
                                      influence_files['forward'], 
                                      influence_files['backward'])
    
    all_nodes = forward_nodes.union(backward_nodes)
    
    if not all_nodes:
        return None
    
    for _, node in nodes_df.iterrows():
        if node['pc'] in all_nodes:
            node_type = []
            if node['pc'] in forward_nodes:
                node_type.append('forward')
            if node['pc'] in backward_nodes:
                node_type.append('backward')
        
            G.add_node(node['pc'], 
                      opcode=node['opcode'],
                      dst=node['dst'] if node['dst'] else "null",
                      src=node['src'] if node['src'] else "null",
                      offset=node['offset'] if node['offset'] else "null",
                      imm=node['imm'] if node['imm'] else "null",
                      syscall=syscall_name,
                      influence_type=','.join(node_type))
    
    for _, edge in edges_df.iterrows():
        if edge['src_pc'] in all_nodes and edge['dst_pc'] in all_nodes:
            G.add_edge(edge['src_pc'], edge['dst_pc'], type=edge['edge_type'])

    return G


def save_graph_as_json(G, filename):
    try:
        data = nx.node_link_data(G)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        pass


def save_graph_as_pdf(G, filename, title):
    try:
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        node_colors = []
        for node in G.nodes():
            influence_type = G.nodes[node].get('influence_type', '')
            if 'forward,backward' in influence_type:
                node_colors.append('purple')
            elif 'forward' in influence_type:
                node_colors.append('blue')
            elif 'backward' in influence_type:
                node_colors.append('red')
            else:
                node_colors.append('gray')
        
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
        
        labels = {node: f"{node}\n{G.nodes[node].get('opcode', '')}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Both',
                      markerfacecolor='purple', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Forward',
                      markerfacecolor='blue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Backward',
                      markerfacecolor='red', markersize=10)
        ]
        plt.legend(handles=legend_elements)
        
        plt.title(title)
        plt.axis('off')
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
    except Exception as e:
        pass

def combine_graphs(facts_dir="souffle/facts", results_dir="souffle/results", output_dir="output"):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    syscall_pairs = read_syscall_results(results_dir)
    
    if not syscall_pairs:
        return None, None
    
    syscall_graphs = {}
    
    for syscall_name, influence_files in syscall_pairs.items():
        
        G = build_syscall_graph(facts_dir, results_dir, syscall_name, influence_files)
        
        if G is None or G.number_of_nodes() == 0:
            continue
            
        syscall_graphs[syscall_name] = G
        
        json_path = os.path.join(output_dir, f"{syscall_name}_graph.json")
        save_graph_as_json(G, json_path)

    if not syscall_graphs:
        return None, None
    
    combined_graph = nx.DiGraph()
    for G in syscall_graphs.values():
        combined_graph.add_nodes_from(G.nodes(data=True))
        combined_graph.add_edges_from(G.edges(data=True))
    
    if combined_graph.number_of_nodes() > 0:
        combined_json_path = os.path.join(output_dir, "combined_graph.json")
        save_graph_as_json(combined_graph, combined_json_path)
    
    return syscall_graphs, combined_graph


if __name__ == "__main__":
    syscall_graphs, combined_graph = combine_graphs()