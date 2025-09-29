import os
import json
import time
from sym.vm import VM
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor


def run_sym(main_dir='output', error_log='sym_log.txt'):

    if not os.path.exists(main_dir):
        return

    with open(error_log, "w") as log_file:
        log_file.write("Error logs for simulation execution: \n")

    sym_overhead = []

    for subfolder in os.listdir(main_dir):

        subfolder_path = os.path.join(main_dir, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        graph_file_path = os.path.join(subfolder_path, "combined_graph.json")
        if not os.path.exists(graph_file_path):
            continue

        # record time
        start_time = time.time()

        vm = VM(file_dir=graph_file_path)
        vm.execute()

        node_states = vm.node_states

        end_time = time.time()
        overhead_time = end_time - start_time
        sym_overhead.append(overhead_time)

        output_dir = os.path.join(subfolder_path, 'runtime.json')
        with open(output_dir, 'w') as fw:
            json.dump(node_states, fw, indent=2)

    with open('ablation/sym_overhead.txt', 'w') as f:
        for overhead_time in sym_overhead:
            f.write(str(overhead_time) + '\n')

    print('Done!')


def process_subfolder(subfolder_path, error_log):
    try:
        graph_file_path = os.path.join(subfolder_path, "combined_graph.json")
        if not os.path.exists(graph_file_path):
            return None

        output_dir = os.path.join(subfolder_path, 'runtime.json')
        if os.path.exists(output_dir):
            return None

        start_time = time.time()

        vm = VM(file_dir=graph_file_path)
        vm.execute()

        node_states = vm.node_states

        end_time = time.time()
        overhead_time = end_time - start_time

        with open(output_dir, 'w') as fw:
            json.dump(node_states, fw, indent=2)

        return overhead_time

    except Exception as e:
        error_message = f"Processing {subfolder_path} error: {str(e)}\n"
        with open(error_log, "a") as log_file:
            log_file.write(error_message)
        return None


def run_sym_online(main_dir, error_log, max_workers, timeout):

    if not os.path.exists(main_dir):
        return

    with open(error_log, "w") as log_file:
        log_file.write("Error logs for simulation execution: \n")

    sym_overhead = []

    subfolders = [
        os.path.join(main_dir, subfolder)
        for subfolder in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, subfolder))
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_subfolder = {
            executor.submit(process_subfolder, subfolder_path, error_log): subfolder_path
            for subfolder_path in subfolders
        }

        for future in as_completed(future_to_subfolder):
            subfolder_path = future_to_subfolder[future]
            try:
                overhead_time = future.result(timeout=timeout)
                if overhead_time is not None:
                    sym_overhead.append(overhead_time)
            except TimeoutError:
                error_message = f"Subfolder {subfolder_path} timeout {timeout}\n"
                with open(error_log, "a") as log_file:
                    log_file.write(error_message)
                print(error_message)
            except Exception as e:
                pass

    with open('sym_overhead_online.txt', 'w') as f:
        for overhead_time in sym_overhead:
            f.write(str(overhead_time) + '\n')

    print('Done!')

