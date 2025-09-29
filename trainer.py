import os
import math
import random
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax, add_self_loops, degree, k_hop_subgraph
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch_geometric.data import Batch
import torch.nn.functional as F

import logging
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def trainer(model, train_dataset, test_dataset, file_label_maps, args):
    if args.test_only == False:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_metric = 0  
    if args.test_mode == 'common':
        best_model_path = f"saved_models/best_model_{args.vul_type}.pth"  
    else:
        best_model_path = f"saved_models/best_model_online_contracts_{args.vul_type}.pth"

    best_train_loss = float('inf') 
    global_step = 0  
    save_every_steps = 5  

    if args.test_only == False:  

        # Time record
        train_start_time = time.time()

        # Training loop
        num_epochs = args.epochs
        for epoch in range(1, num_epochs + 1):
            model.train()

            total_loss = 0

            for batch_idx, batch in enumerate(train_dataloader):

                global_step += 1

                data = batch.to(args.device)
                optimizer.zero_grad()

                labels = data.y

                graph_emb, graph_emb_with_prefix, logits, prefix = model(data)

                loss = criterion(logits, labels)

                alpha = 0.5 

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if global_step % save_every_steps == 0:
                    avg_train_loss = total_loss / (batch_idx + 1)  
                    print(f"Step {global_step}, Avg Train Loss: {avg_train_loss:.4f}")

                    if avg_train_loss < best_train_loss:
                        best_train_loss = avg_train_loss
                        torch.save(model.state_dict(), best_model_path)

        train_end_time = time.time()
        avg_train_time = train_end_time - train_start_time

        print(f"Training complete. It took {avg_train_time:.4f} seconds.")

    if args.test_mode == 'online_train':
        exit()

    model.load_state_dict(torch.load(best_model_path))
    model.to(args.device)

    print(f'Loading model from {best_model_path} finished. Testing .... ')

    test_start_time = time.time()
    
    val = do_test(model, test_dataloader, criterion, file_label_maps, args.device, args.vul_type,
                  args.test_mode, args.cur_part, args.cur_inner_part)

    test_end_time = time.time()
    avg_test_time = test_end_time - test_start_time

    if args.test_mode == 'common':
        print(f"Final Evaluation - Test Loss: {val['loss']:.4f}, Acc: {val['accuracy']:.4f}, "
              f"P: {val['precision']:.4f}, R: {val['recall']:.4f}, F1: {val['f1']:.4f}, "
              f"Macro-P: {val['macro_precision']:.4f}, Macro-R: {val['macro_recall']:.4f}, Macro-F1: {val['macro_f1']:.4f}")

    print(f"Test spends {avg_test_time:.4f} seconds.")


def do_test(model, loader, criterion, file_label_maps, device, vul_type, test_mode, cur_part, cur_inner_part):
    model.eval()

    total_loss = 0
    all_preds = []  
    all_labels = [] 
    all_filenames = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, _, out, _ = model(batch) 
            labels = batch.y  
            loss = criterion(out, labels)  
            total_loss += loss.item()

            pred = out.argmax(dim=1)  
            all_preds.extend(pred.cpu().tolist()) 
            all_labels.extend(labels.cpu().tolist())  

            labeled_filenames = batch.file_label.detach().cpu().tolist()
            filenames = [file_label_maps[i] for i in labeled_filenames]

            all_filenames.extend(filenames)

    if test_mode == 'common':

        with open(f'Pred_True_{vul_type}.txt', 'w') as fw:
            for pred, true, filename in zip(all_preds, all_labels, all_filenames):
                line = f"Pred: {pred} , True: {true} ; File: {filename}\n"
                fw.write(line)

        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        accuracy = accuracy_score(all_labels, all_preds)

        macro_precision = precision_score(all_labels, all_preds, average='macro')
        macro_recall = recall_score(all_labels, all_preds, average='macro')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        all_precision = precision_score(all_labels, all_preds, average=None)
        all_recall = recall_score(all_labels, all_preds, average=None)
        all_f1 = f1_score(all_labels, all_preds, average=None)

        print(all_precision)
        print('\n')
        print(all_recall)
        print('\n')
        print(all_f1)
        print('\n')

        avg_loss = total_loss / len(loader)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1
        }

    elif test_mode == 'online_test':
        with open(f'./online_pred_results/Pred_Online_{vul_type}_{cur_part}_{cur_inner_part}.txt', 'w') as fw:
            for pred, filename in zip(all_preds, all_filenames):
                line = f"Pred: {pred}; File: {filename}\n"
                fw.write(line)

        print(f'All results are saved into online_pred_results/Pred_Online_{vul_type}_{cur_part}_{cur_inner_part}.txt')

