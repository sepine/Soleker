import json
import os
import numpy as np
import random
import pickle

import torch
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import train_test_split
from datetime import datetime
from collections import defaultdict
import time

import orjson

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


class BPFDataset(Dataset):
    def __init__(self, dataflow_graphs, node_to_idx, edge_index_list, edge_attr_list, keys, glove_embeds, args):

        super(BPFDataset, self).__init__()

        self.args = args

        self.dataflow_graphs = dataflow_graphs
        self.node_to_idx = node_to_idx
        self.edge_index_list = edge_index_list
        self.edge_attr_list = edge_attr_list
        self.keys = keys
        self.glove_embeds = glove_embeds

        self.dataset = self.convert_all_to_pyg_data()

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]

    def convert_all_to_pyg_data(self):
        dataset = []
        for i, dfg in enumerate(self.dataflow_graphs):
            edge_index = self.edge_index_list[i]
            edge_attr = self.edge_attr_list[i]

            num_nodes = edge_index.max().item() + 1
            x = torch.zeros((num_nodes, self.args.input_dim), dtype=torch.float)
            for node_id in range(num_nodes):

                node_feature = str(dfg[node_id]) 
                x[node_id] = self.get_glove_embedding(node_feature)

            label = self.numeric_label(self.keys[i])
            y = torch.tensor([label], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            dataset.append(data)
        return dataset

    def add_node_features(self):
        opcodes = set()
        for dfg in self.dataflow_graphs:
            for node in dfg.get('nodes', []):
                opcodes.add(node['opcode'])  
        opcode_to_idx = {opcode: idx for idx, opcode in enumerate(sorted(opcodes))}
        num_opcodes = len(opcode_to_idx)

        for i, data in enumerate(self.dataset):
            x = torch.zeros((len(self.node_to_idx), num_opcodes), dtype=torch.float)
           
            for node in self.dataflow_graphs[i].get('nodes', []):
                pc = node['pc']
                opcode = node['opcode']
                node_idx = self.node_to_idx.get(pc)
                if node_idx is not None and opcode in opcode_to_idx:
                    opcode_idx = opcode_to_idx[opcode]
                    x[node_idx, opcode_idx] = 1.0
            data.x = x

    def numeric_label(self, label_str):
        if 'fixed' in label_str:
            return 0
        else:
            if 'acpi' in label_str:
                return 1
            elif 'mkc' in label_str:
                return 2
            elif 'moc' in label_str:
                return 3
            elif 'msc' in label_str:
                return 4


def load_all_dfgs(json_file_path):

    dfg_json_files = []
    try:
        for filename in os.listdir(json_file_path):
            if filename.endswith(".json"): 
                with open(os.path.join(json_file_path, filename), 'r') as f:
                    dfg_json_files.append(json.load(f))
        print(f"Successfully loaded {len(dfg_json_files)} DFGs from {json_file_path}")
        return dfg_json_files
    except Exception as e:
        print(f"Error decoding JSON: {e}")
        raise


def load_all_dfgs(args):
    json_file_path = args.dataset_path

    dfg_json_files = {}
    runtime_json_files = {}

    try:
        for subfolder in os.listdir(json_file_path):

            if args.vul_type not in subfolder:
                continue

            subfolder_path = os.path.join(json_file_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            file_path = os.path.join(subfolder_path, 'combined_graph.json')
            if not os.path.exists(file_path):
                continue

            with open(file_path, 'r') as f:
                dfg_json_files[subfolder] = json.load(f)

            runtime_file_path = os.path.join(subfolder_path, 'runtime.json')
            with open(runtime_file_path, 'r') as f:
                runtime_json_files[subfolder] = json.load(f)

        return dfg_json_files, runtime_json_files

    except Exception as e:
        print(f"Error decoding JSON: {e}")
        raise


def load_online_dfgs(args):

    json_file_path = args.online_dataset_path

    dfg_json_files = {}
    runtime_json_files = {}

    subfolders = []
    with open('online_contracts.txt', 'r') as fr:
        for line in fr.readlines():
            subfolders.append(line.strip())

    for subfolder in subfolders:

        subfolder_path = os.path.join(json_file_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        file_path = os.path.join(subfolder_path, 'combined_graph.json')
        if not os.path.exists(file_path):
            continue

        try:
            with open(file_path, 'r') as f:
                dfg_json_files[subfolder] = json.load(f)

            runtime_file_path = os.path.join(subfolder_path, 'runtime.json')
            with open(runtime_file_path, 'r') as f:
                runtime_json_files[subfolder] = json.load(f)

        except Exception as e:
            print(f"Error decoding JSON: {e}")
            continue

    print(f"Successfully loaded {len(dfg_json_files)} DFGs from {json_file_path}")

    return dfg_json_files, runtime_json_files


def parse_graph_to_pyg(graph_dict, runtime_dict, doc2vec, embeds, args):
    pyg_graphs = []

    prefix_label = []
    overhead_prefix = []

    file_label_maps = {}
    label_idx = 0

    edge_type_mapping = {
        "reg": 1,
        "jump": 2,
        "return": 3,
        "call": 4,
    }

    num_instructions = 69

    for key, graph in tqdm(graph_dict.items(), desc="Parsing graphs"):

        start_time = time.time()

        if key not in runtime_dict:
            continue

        runtime_node_infos = runtime_dict[key]

        nodes = graph["nodes"]  
        node_features = []

        node_ids = {}  

        node_id_index = 0
        instruction_freq = [0] * num_instructions  

        ldxb_flag = 0
        last_node_id = -1

        op_list = []

        arbitrary_cpi_key_flag = 0

        missing_key_flag = 0

        for idx, node in enumerate(nodes):
            opcode = node["opcode"] 
            node_id = node["id"]
            if args.vul_type == 'msc':
                if opcode == 'ldxb':
                    offset = node["offset"]
                    if offset.endswith('8'):
                        last_node_id = node_id
                elif opcode == 'jeq' or opcode == 'jne':
                    imm = node["imm"]
                    if imm == '0' and last_node_id + 1 == node_id:
                        ldxb_flag += 1
            elif args.vul_type == 'moc':
                if opcode == 'ldxdw' and len(op_list) == 0:
                    offset = node["offset"]
                    if offset in ['0x18', '0x48', '0x78', '0xa8', '0xd8', '0x108', '0x138', '0x168']:
                        op_list.append((opcode, None, None))
                        last_node_id = node_id
                elif len(op_list) > 0:
                    if opcode == 'mov64' and len(op_list) == 1 and node_id > last_node_id:

                        imm = node["imm"]
                        if imm == '0':
                           
                            try:
                                next_1 = nodes[idx + 1]
                                if next_1['opcode'] == 'ldxdw' and next_1['offset'] == '0x18':
                                    next_2 = nodes[idx + 2]
                                    if next_2['opcode'] == 'ldxdw' and next_2['offset'] == '0x18':
                                        next_3 = nodes[idx + 3]
                                        if next_3['opcode'] in ['jne', 'jeq']:
                                            n1_dst = next_1['dst']
                                            n2_dst = next_2['dst']
                                            n3_dst = next_3['dst']
                                            n3_src = next_3['src']

                                            if n2_dst == n3_dst and n3_src == n1_dst:
                                                ldxb_flag += 1
                            except Exception as e:
                                pass
            elif args.vul_type == 'mkc':
                if opcode == 'ldxdw' and len(op_list) == 0 and node['offset'] == '0x0':
                    dst = node['dst']
                    try:
                        next = nodes[idx + 1]
                        if next['opcode'] == 'ldxdw' and next['offset'] == '0x0' and next['src'] == dst:
                            op_list.append((opcode, dst, None))
                            last_node_id = next['id']

                            next_next = nodes[idx + 2]
                            if next_next['opcode'] == 'lddw':
                                next_next_next = nodes[idx + 3]
                                if next_next_next['opcode'] == 'jeq' and next_next_next['src'] == next_next['dst'] and next_next_next['dst'] == next['dst']:
                                    missing_key_flag = 1  
                                    last_node_id = next_next['id']
                    except Exception as e:
                        pass
                elif opcode == 'mov64' and missing_key_flag == 1 and len(op_list) > 0 and node_id > last_node_id:
                    imm = node["imm"]
                    if imm == '0':
                        last_dst = op_list[-1][1]
                        try:
                            next_1 = nodes[idx + 1]
                            if next_1['opcode'] == 'ldxdw' and next_1['offset'] == '0x18' and last_dst == next_1['src']:
                                next_2 = nodes[idx + 2]
                                if next_2['opcode'] == 'lddw':
                                    next_3 = nodes[idx + 3]
                                    if next_3['opcode'] in ['jne', 'jeq']:
                                        n1_dst = next_1['dst']
                                        n2_dst = next_2['dst']
                                        n3_dst = next_3['dst']
                                        n3_src = next_3['src']
                                        if n2_dst == n3_src and n3_dst == n1_dst:
                                            ldxb_flag += 1
                        except Exception as e:
                            pass
            elif args.vul_type == 'acpi':
                if opcode == 'ldxdw' and len(op_list) == 0 and node['offset'] == '0x0':
                    dst = node['dst']
                    try:
                        next = nodes[idx + 1]
                        if next['opcode'] == 'ldxdw' and next['offset'] == '0x0' and next['src'] == dst:

                            op_list.append((opcode, dst, None))
                            last_node_id = next['id']

                            next_next = nodes[idx + 2]
                            if next_next['opcode'] == 'lddw':
                                arbitrary_cpi_key_flag = 1 
                                last_node_id = next_next['id']
                            elif next_next['opcode'] == 'jeq' and next_next['imm'] == '0' and next_next['dst'] == next['dst']:
                                arbitrary_cpi_key_flag = 2  
                                last_node_id = next_next['id']

                    except Exception as e:
                        pass
                elif len(op_list) > 0:
                    if arbitrary_cpi_key_flag == 1:   
                        if opcode == 'mov64' and len(op_list) == 1 and node_id > last_node_id:
                            imm = node["imm"]
                            if imm == '0':
                                last_dst = op_list[-1][1]
                                try:
                                    next_1 = nodes[idx + 1]
                                    if next_1['opcode'] == 'ldxdw' and next_1['offset'] == '0x18' and last_dst == next_1['src']:
                                        next_2 = nodes[idx + 2]
                                        if next_2['opcode'] == 'lddw':
                                            next_3 = nodes[idx + 3]
                                            if next_3['opcode'] in ['jne', 'jeq']:
                                                n1_dst = next_1['dst']
                                                n2_dst = next_2['dst']
                                                n3_dst = next_3['dst']
                                                n3_src = next_3['src']
                                                if n2_dst == n3_src and n3_dst == n1_dst:
                                                    ldxb_flag += 1
                                except Exception as e:
                                    pass
                    elif arbitrary_cpi_key_flag == 2:  
                        if opcode == 'mov64' and len(op_list) == 1 and node_id > last_node_id:

                            imm = node["imm"]
                            if imm == '0':
                                last_dst = op_list[-1][1]
                                try:
                                    next_1 = nodes[idx + 1]
                                    if next_1['opcode'] == 'ldxdw' and next_1['offset'] == '0x18' and last_dst == \
                                            next_1['src']:
                                        next_2 = nodes[idx + 2]
                                        if next_2['opcode'] == 'ldxdw' and next_2['src'] == 'r10':
                                            next_3 = nodes[idx + 3]
                                            if next_3['opcode'] in ['jne', 'jeq']:
                                                n1_dst = next_1['dst']
                                                n2_dst = next_2['dst']
                                                n3_dst = next_3['dst']
                                                n3_src = next_3['src']
                                                if n2_dst == n3_src and n3_dst == n1_dst:
                                                    ldxb_flag += 1
                                        elif next_2['opcode'] == 'jne' and next_2['imm'] == '0' and next_2['dst'] == next_1['dst']:
                                            ldxb_flag += 1
                                except Exception as e:
                                    pass

        
            node_ids[node_id] = node_id_index  
            node_id_index += 1

            if str(node_id) in runtime_node_infos:

                nodes_runtime = runtime_node_infos[str(node_id)]
                nodes_runtime_exec = nodes_runtime['exec']
                nodes_runtime_exec = list(set(nodes_runtime_exec))

                node_emb = []
                for n_r in nodes_runtime_exec:
                    if n_r in doc2vec:
                        embed = doc2vec[n_r]
                        node_emb.append(embed)
                    else:
                        raise Exception(f"Unknown runtime exec tokens: {n_r}")

                nodes_runtime_memory = nodes_runtime['memory']
                for k, v in nodes_runtime_memory.items():
                    k_v = f'{k} = {v}'
                    if k_v in doc2vec:
                        embed = doc2vec[k_v]
                        node_emb.append(embed)
                    else:
                        raise Exception(f"Unknown runtime memory tokens: {k_v}")

                node_emb = np.mean(node_emb, axis=0)

            else:  
                if opcode in embeds:
                    node_emb = embeds[opcode]
                else:
                    node_emb = np.zeros(args.input_dim, dtype=np.float32)

            node_features.append(node_emb)

            opcode_idx = hash(opcode) % num_instructions 
            instruction_freq[opcode_idx] += 1

        instruction_freq = np.array(instruction_freq, dtype=np.float32)
        total_count = instruction_freq.sum()
        if total_count > 0:
            instruction_prob_dist = instruction_freq / total_count  
        else:
            instruction_prob_dist = instruction_freq 

        instruction_prob_dist = torch.tensor(instruction_prob_dist, dtype=torch.float)

        node_features = torch.tensor(np.array(node_features), dtype=torch.float)

        links = graph["links"]  
        edge_index = []
        edge_attr = []

        for link in links:
            source = link["source"]
            target = link["target"]
            edge_type = link["type"]  

            if source in node_ids and target in node_ids:
                src_idx = node_ids[source]
                tgt_idx = node_ids[target]

                edge_index.append((src_idx, tgt_idx))

                for pre_ed_ty in edge_type_mapping:
                    if pre_ed_ty in edge_type:
                        edge_attr.append(edge_type_mapping[pre_ed_ty])
                        break
                else:
                    raise ValueError(f"Unknown edge type: {edge_type}")

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  

        num_edge_types = len(edge_type_mapping)
        edge_attr_one_hot = torch.zeros((len(edge_attr), num_edge_types), dtype=torch.float)  

        for i, edge_type_idx in enumerate(edge_attr):
            edge_attr_one_hot[i, edge_type_idx] = 1.0

        edge_attr = edge_attr_one_hot

        y = torch.tensor([numeric_label(key)], dtype=torch.long)

        file_label_maps[label_idx] = key
        file_label = torch.tensor([label_idx], dtype=torch.long)

        ldxb_flag = torch.tensor([ldxb_flag], dtype=torch.float)

        pyg_graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y,
                         prob_dist=instruction_prob_dist, file_label=file_label, ldxb_flag=ldxb_flag)
        pyg_graphs.append(pyg_graph)

        label_idx = label_idx + 1

    if args.test_mode == 'common':
        with open(f"file_label_maps_{args.vul_type}.json", "w") as f:
            json.dump(file_label_maps, f, indent=2)

    print(f"Successfully parse graphs into PyG format.")

    with open('./cache/pygs.pkl', 'wb') as fw:
        pickle.dump(pyg_graphs, fw)

    return pyg_graphs, file_label_maps


def numeric_label(label_str):
    if 'fixed' in label_str:
        return 0
    else:
        return 1


def split_graphs(graph_list, test_ratio=0.2, seed=42):
    train_graphs, test_graphs = train_test_split(
        graph_list, test_size=test_ratio, random_state=seed
    )
    return train_graphs, test_graphs


def load_doc2vec_from_json_dict(file_path):
    with open(file_path, "r") as f:
        # feature_dict = json.load(f)
        feature_dict = orjson.loads(f.read())
    feature_dict = {inst: np.array(vector) for inst, vector in feature_dict.items()}
    return feature_dict


def load_bpf_datasets(args):

    start_time = datetime.now()

    glove_embeds = load_glove_embeddings(args.glove_path)

    if args.test_mode == 'common':

        cache_path = f'./cache/data_{args.test_mode}_{args.vul_type}.pkl'
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                train_dataset, test_dataset, file_label_maps = pickle.load(cache_file)

        else:
            dataflow_graphs, runtime_json_files = load_all_dfgs(args)

            # Load pre-trained Doc2Vec model
            doc2vec = load_doc2vec_from_json_dict(file_path=f'node_features.json')

            pyg_graphs, file_label_maps = parse_graph_to_pyg(dataflow_graphs, runtime_json_files, doc2vec, glove_embeds, args)

            train_dataset, test_dataset = split_graphs(pyg_graphs, test_ratio=0.2, seed=42)

            with open(cache_path, 'wb') as cache_file:
                pickle.dump((train_dataset, test_dataset, file_label_maps), cache_file)

        logger.info('Build Train/Test finished! The total number is %d/%d' % (len(train_dataset), len(test_dataset)))

        end_time = datetime.now()
        logger.info('Load dataset spends %s (s)' % str((end_time - start_time).seconds))

        return train_dataset, test_dataset, file_label_maps

    elif args.test_mode == 'online_train': 
        dataflow_graphs, runtime_json_files = load_all_dfgs(args)
        doc2vec = load_doc2vec_from_json_dict(file_path=f'node_features.json')
        train_pyg_graphs, file_label_maps = parse_graph_to_pyg(dataflow_graphs, runtime_json_files, doc2vec, glove_embeds, args)

        return train_pyg_graphs, None, None

    elif args.test_mode == 'online_test':

        online_data_flow_graphs, online_runtime_json_files = load_online_dfgs(args)

        online_doc2vec = load_doc2vec_from_json_dict(
            file_path=f'../node_features_online_contracts_{args.cur_part}.json')

        print('Loading the doc2vec json file successfully. ')

        print('Start building PyG test dataset... ')

        online_pyg_graphs, online_file_label_maps = parse_graph_to_pyg(online_data_flow_graphs,
                                                                        online_runtime_json_files,
                                                                        online_doc2vec,
                                                                        glove_embeds,
                                                                        args)

        print(len(online_pyg_graphs), len(online_file_label_maps))

        logger.info(
            'Build Online Test finished! The total number is %d' % len(online_pyg_graphs))

        end_time = datetime.now()
        logger.info('Load dataset spends %s (s)' % str((end_time - start_time).seconds))

        return None, online_pyg_graphs, online_file_label_maps


def is_pyg_graph_empty(pyg_graph):
    if pyg_graph is None:
        return True
    if pyg_graph.x is None or pyg_graph.edge_index is None:
        return True
    if pyg_graph.x.size(0) == 0 or pyg_graph.edge_index.size(1) == 0:
        return True
    return False


def load_glove_embeddings(glove_path):
    """
    Load pre-trained GloVe embeddings from a file.
    """
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    print(f"Loaded {len(embeddings)} GloVe embeddings.")
    return embeddings