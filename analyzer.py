import os
from souffle.fact_generator import generate_souffle_facts
from souffle.graph import combine_graphs
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import traceback

import warnings
warnings.filterwarnings("ignore")


def combine(main_dir='souffle/results', error_log='combine_error_log.txt'):

    if not os.path.exists(main_dir):
        return

    with open(error_log, "w") as log_file:
        log_file.write("Error logs for combining graphs: \n")

    for subfolder in os.listdir(main_dir):
        subfolder_path = os.path.join(main_dir, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        cfg_file_path = os.path.join(subfolder_path, "cfg.json")
        if not os.path.exists(cfg_file_path):
            continue

        facts_folder_path = os.path.join(subfolder_path, "facts")
        results_path = os.path.join(subfolder_path, "output")

        output_dir = os.path.join('output', subfolder)

        try:
            combine_graphs(facts_dir=facts_folder_path, results_dir=results_path, output_dir=output_dir)

        except Exception as e:
            with open(error_log, "a") as log_file:
                log_file.write(f"{subfolder_path}: {e}\n")

    print("Finished")


def process_subfolder(main_dir, subfolder, error_log):
    subfolder_path = os.path.join(main_dir, subfolder)

    cfg_file_path = os.path.join(subfolder_path, "cfg.json")
    if not os.path.exists(cfg_file_path):
        return

    combine_graph_paths = os.path.join("./online_output", subfolder)
    combine_graph_paths = os.path.join(combine_graph_paths, "combined_graph.json")
    if os.path.exists(combine_graph_paths):
        return

    facts_folder_path = os.path.join(subfolder_path, "facts")
    results_path = os.path.join(subfolder_path, "output")

    result_output_dir = os.path.join('./online_output', subfolder)

    try:
        combine_graphs(facts_dir=facts_folder_path, results_dir=results_path, output_dir=result_output_dir)

    except Exception as e:
        with open(error_log, "a") as log_file:
            log_file.write(f"{subfolder_path}: {e}\n")


def combine_online_contracts(main_dir, error_log, max_workers, timeout):

    if not os.path.exists(main_dir):
        return

    with open(error_log, "w") as log_file:
        log_file.write("Error logs for combining graphs: \n")

    subfolders = [
        subfolder for subfolder in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, subfolder))
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_subfolder, main_dir, subfolder, error_log): subfolder
            for subfolder in subfolders
        }

        for future in as_completed(futures):
            subfolder = futures[future]
            try:
                future.result(timeout=timeout)
            except TimeoutError:
                with open(error_log, "a") as log_file:
                    log_file.write(f"{subfolder}: Timeout\n")
            except Exception as e:
                with open(error_log, "a") as log_file:
                    log_file.write(f"{subfolder}: {e}\n")

