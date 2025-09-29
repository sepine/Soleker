from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import nltk
import os
import json
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from gensim.models.callbacks import CallbackAny2Vec
import time

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

nltk.download('punkt')
nltk.download('punkt_tab')


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.start_time = None

    def on_epoch_begin(self, model):
        self.start_time = time.time()
        print(f"Epoch {self.epoch + 1} start ...")

    def on_epoch_end(self, model):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"Epoch {self.epoch + 1} end. Time taken: {elapsed_time:.2f} seconds.")
        self.epoch += 1


def load_all_dfgs(vul_type):
    json_file_path = './output'

    instructions = []

    try:
        for subfolder in os.listdir(json_file_path):

            if vul_type not in subfolder:
                continue

            subfolder_path = os.path.join(json_file_path, subfolder)

            if not os.path.isdir(subfolder_path):
                continue

            file_path = os.path.join(subfolder_path, 'runtime.json')
            if not os.path.exists(file_path):
                continue

            with open(file_path, 'r') as f:
                runtime_node_json = json.load(f)

            for pc, node in runtime_node_json.items():
                execs = node['exec']
                for exec in execs:
                    instructions.append(exec)

                memory = node['memory']

                for k, v in memory.items():
                    instructions.append(f"{k} = {v}")

        print(f"Successfully loaded {len(instructions)} runtimes from {json_file_path}")

        instructions = list(set(instructions))
        print(f"Retained {len(instructions)} unique instructions")

        return instructions

    except Exception as e:
        print(f"Error decoding JSON: {e}")
        raise


def preprocess_instruction(instruction):
    return nltk.word_tokenize(instruction.lower())  



instructions = load_all_dfgs(vul_type="")


tagged_instructions = [TaggedDocument(words=preprocess_instruction(inst), tags=[str(idx)]) for idx, inst in enumerate(instructions)]

model = Doc2Vec(
    vector_size=300,  
    window=3,        
    min_count=1,    
    workers=10,       
    epochs=50,       
    dm=1,           
)

model.build_vocab(tagged_instructions)

print("Training Doc2Vec model...")
model.train(tagged_instructions, total_examples=model.corpus_count, epochs=model.epochs)


print("Training complete!")


model.save(f"doc2vec")
print(f"Model saved to 'doc2vec'.")


model = Doc2Vec.load(f"doc2vec")


def instruction_to_vector(instruction):
    tokenized = preprocess_instruction(instruction)
    vector = model.infer_vector(tokenized)
    return vector

node_features = [instruction_to_vector(inst) for inst in instructions]

print('Convert node features finished. ')


def save_as_json_dict(instructions, node_features, file_path=f"node_features.json"):
    feature_dict = {inst: vector.tolist() for inst, vector in zip(instructions, node_features)}
    with open(file_path, "w") as f:
        json.dump(feature_dict, f, indent=2)
    print(f"Instructions and features saved to {file_path}.")

