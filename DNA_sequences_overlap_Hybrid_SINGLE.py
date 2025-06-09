import random
import numpy as np
import multiprocessing
from multiprocessing import Manager, Lock
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt

ALPHA_INIT = 1.0
BETA_INIT = 2.0
RHO_INIT = 0.1
Q0_INIT = 0.9
PHI = 0.1
Q = 100
NUM_ANTS = 20
NUM_ITER = 20
TAU0 = 0.1
COOP_INTERVAL = 3


def overlap(a, b, min_length=20):
    max_len = min(len(a), len(b))
    for i in range(max_len, min_length - 1, -1):
        if a[-i:] == b[:i]:
            return i
    return 0


def read_dna_from_file(filename):
    with open(filename, 'r') as f:
        content = f.read().strip()
    lines = content.split('\n')
    dna_sequence = ""
    for line in lines:
        if not line.startswith('>') and not line.startswith('#'):
            dna_sequence += line.strip().upper()
    return dna_sequence


def generate_reads_from_dna(dna, read_length=150, min_overlap=30, max_overlap=120, shuffle=True):
    reads = []
    i = 0
    while i + read_length <= len(dna):
        reads.append(dna[i:i + read_length])
        random_overlap = random.randint(min_overlap, max_overlap)
        step = read_length - random_overlap
        i += step
    if i < len(dna):
        reads.append(dna[-read_length:])
    if shuffle:
        random.shuffle(reads)
    return reads


def get_input_filename():
    while True:
        filename = input("Enter filename: ").strip()
        if filename:
            if os.path.exists(filename):
                return filename
            else:
                print(f"File '{filename}' not found.")

def construct_path_acs(pheromone, heuristic, Q0, BETA, ALPHA):
    n = heuristic.shape[0]
    path = []
    visited = set()
    current = random.randint(0, n - 1)
    path.append(current)
    visited.add(current)

    while len(visited) < n:
        candidates = [j for j in range(n) if j not in visited]
        if not candidates:
            break
        q = random.random()
        if q <= Q0:
            best_j, best_value = None, -1
            for j in candidates:
                tau = pheromone[current][j]
                eta = heuristic[current][j]
                value = tau * (eta ** BETA)
                if value > best_value:
                    best_value = value
                    best_j = j
            next_node = best_j if best_j is not None else random.choice(candidates)
        else:
            probs = []
            total = 0
            for j in candidates:
                tau = pheromone[current][j] ** ALPHA
                eta = heuristic[current][j] ** BETA
                prob = tau * eta
                probs.append(prob)
                total += prob
            if total == 0:
                next_node = random.choice(candidates)
            else:
                probs = [p / total for p in probs]
                next_node = np.random.choice(candidates, p=probs)

        pheromone[current][next_node] = (1 - PHI) * pheromone[current][next_node] + PHI * TAU0
        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return path


def path_length(reads, path):
    total = len(reads[path[0]])
    for i in range(1, len(path)):
        olap = overlap(reads[path[i - 1]], reads[path[i]])
        total += len(reads[path[i]]) - olap
    return total


def global_pheromone_update(pheromone, path, length, RHO):
    pheromone *= (1 - RHO)
    deposit = Q / length
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        pheromone[a][b] += RHO * deposit


def build_heuristic(reads):
    n = len(reads)
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic[i][j] = overlap(reads[i], reads[j])
    max_h = np.max(heuristic)
    if max_h > 0:
        heuristic = heuristic / max_h
    return heuristic


def build_sequence(reads, path):
    sequence = reads[path[0]]
    for i in range(1, len(path)):
        olap = overlap(sequence, reads[path[i]])
        sequence += reads[path[i]][olap:]
    return sequence


def run_colony_with_tracking(colony_id, reads, heuristic, shared_best, lock, result_list, performance_tracker):
    n = len(reads)
    pheromone = np.full((n, n), TAU0)
    best_path = None
    best_length = float('inf')

    colony_performance = []

    for it in range(NUM_ITER):
        Q0 = min(0.9, Q0_INIT + it * 0.05)
        BETA = min(5.0, BETA_INIT + it * 0.3)
        RHO = max(0.05, RHO_INIT - it * 0.005)
        ALPHA = ALPHA_INIT

        paths = [construct_path_acs(pheromone, heuristic, Q0, BETA, ALPHA) for _ in range(NUM_ANTS)]
        lengths = [path_length(reads, p) for p in paths]
        idx = np.argmin(lengths)
        current_best_path = paths[idx]
        current_best_length = lengths[idx]

        if current_best_length < best_length:
            best_path = current_best_path
            best_length = current_best_length

        colony_performance.append({
            'iteration': it,
            'best_fitness': min(lengths),
            'worst_fitness': max(lengths),
            'avg_fitness': np.mean(lengths),
            'colony_best': best_length
        })

        global_pheromone_update(pheromone, best_path, best_length, RHO)

        if it % COOP_INTERVAL == 0:
            with lock:
                if shared_best['length'] is None or best_length < shared_best['length']:
                    shared_best['length'] = best_length
                    shared_best['path'] = best_path
                else:
                    global_pheromone_update(pheromone, shared_best['path'], shared_best['length'], RHO)

    result_list[colony_id] = (best_path, best_length)
    performance_tracker[colony_id] = colony_performance


def run_optimization(original_dna, num_colonies, read_length=150, min_overlap=30, max_overlap=120):
    reads = generate_reads_from_dna(original_dna, read_length, min_overlap, max_overlap)
    heuristic = build_heuristic(reads)
    original_length = len(original_dna)

    print(f"Original DNA length: {original_length}")
    print(f"Generated {len(reads)} reads of length {read_length}")

    manager = Manager()
    shared_best = manager.dict()
    shared_best['path'] = None
    shared_best['length'] = None
    lock = Lock()
    result_list = manager.list([None] * num_colonies)
    performance_tracker = manager.dict()

    start_time = time.time()

    jobs = []
    for i in range(num_colonies):
        p = multiprocessing.Process(target=run_colony_with_tracking,
                                    args=(i, reads, heuristic, shared_best, lock,
                                          result_list, performance_tracker))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    optimization_time = time.time() - start_time

    final_best = min(result_list, key=lambda x: x[1])
    final_path = final_best[0]
    final_length = final_best[1]

    assembled = build_sequence(reads, final_path)
    assembled_length = len(assembled)

    print(f"\nOptimization completed in {optimization_time:.2f} seconds")
    print(f"Final assembled length: {assembled_length}")
    print(f"Length difference: {abs(assembled_length - original_length)}")
    print(f"Accuracy: {100 - (abs(assembled_length - original_length) / original_length * 100):.2f}%")

    return {
        'original_length': original_length,
        'assembled_length': assembled_length,
        'length_difference': abs(assembled_length - original_length),
        'optimization_time': optimization_time,
        'assembled_sequence': assembled,
        'original_sequence': original_dna,
        'best_fitness': final_length,
        'performance_data': dict(performance_tracker)
    }


def plot_results(performance_data, original_length, num_colonies):
    iterations = range(NUM_ITER)

    all_best = []
    all_worst = []
    all_avg = []
    colony_bests = []

    for iteration in iterations:
        iter_best = []
        iter_worst = []
        iter_avg = []
        iter_colony_bests = []

        for colony_id in range(num_colonies):
            if colony_id in performance_data:
                data = performance_data[colony_id][iteration]
                iter_best.append(data['best_fitness'])
                iter_worst.append(data['worst_fitness'])
                iter_avg.append(data['avg_fitness'])
                iter_colony_bests.append(data['colony_best'])

        all_best.append(np.mean(iter_best))
        all_worst.append(np.mean(iter_worst))
        all_avg.append(np.mean(iter_avg))
        colony_bests.append(np.min(iter_colony_bests))

    best_accuracy = [abs(fitness - original_length) for fitness in all_best]
    worst_accuracy = [abs(fitness - original_length) for fitness in all_worst]
    avg_accuracy = [abs(fitness - original_length) for fitness in all_avg]
    colony_best_accuracy = [abs(fitness - original_length) for fitness in colony_bests]

    plt.figure(figsize=(14, 8))

    plt.plot(iterations, best_accuracy, 'g-', linewidth=2, label='Best Fitness Accuracy', alpha=0.8)
    plt.plot(iterations, avg_accuracy, 'b-', linewidth=2, label='Average Fitness Accuracy', alpha=0.8)
    plt.plot(iterations, worst_accuracy, 'r-', linewidth=2, label='Worst Fitness Accuracy', alpha=0.8)
    plt.plot(iterations, colony_best_accuracy, 'purple', linewidth=3, label='Global Best Accuracy', alpha=0.9)

    plt.axhline(y=0, color='black', linestyle='--', linewidth=2,
                label='Perfect Accuracy (Original Length)', alpha=0.7)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Length Difference from Original (Accuracy)', fontsize=12)
    plt.title(f'Multi-Colony ACO Performance Over Time\n({num_colonies} Colonies, Original Length: {original_length})',
              fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{timestamp}_colony_performance.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved as: {filename}")
    plt.show()


if __name__ == "__main__":
    filename = get_input_filename()
    original_dna = read_dna_from_file(filename)

    NUM_COLONIES = 4
    available_cores = multiprocessing.cpu_count()
    if NUM_COLONIES > available_cores:
        print(f"Warning: Requested {NUM_COLONIES} colonies but only {available_cores} CPU cores available.")
        print(f"Reducing number of colonies to {available_cores}.")
        NUM_COLONIES = available_cores

    print(f"Using {NUM_COLONIES} colonies on {available_cores} available cores")

    result = run_optimization(original_dna, NUM_COLONIES)

    plot_results(result['performance_data'], result['original_length'], NUM_COLONIES)

    print(f"Final assembled sequence: {result['assembled_sequence'][:200]}...")

# DNA_SEQ_viral_SarsCov2.txt: 34 seconds, perfect assembly
# RNA_SEQ_mouse_Ribosomial45S.txt: 25 seconds, 367 nt different length
# DNA_SEQ_human_BRCA1.txt: 199 seconds, perfect assembly
# DNA_SEQ_human_BRCA1_300k.txt: 2257 seconds, 794 nt difference
