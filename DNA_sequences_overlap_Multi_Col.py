import random
import numpy as np
import multiprocessing
from multiprocessing import Manager, Lock
import time
import Levenshtein

# params
ALPHA = 1.0
BETA = 2.0
RHO = 0.1
PHI = 0.1
Q0 = 0.9
Q = 100
NUM_ANTS = 20
NUM_ITER = 20
TAU0 = 0.1
NUM_COLONIES = 4
COOP_INTERVAL = 3


def overlap(a, b, min_length=20):
    max_len = min(len(a), len(b))
    for i in range(max_len, min_length - 1, -1):
        if a[-i:] == b[:i]:
            return i
    return 0


def random_dna(length):
    return ''.join(random.choices('ACGT', k=length))


def generate_reads_from_dna(dna, read_length=100, min_overlap=10, max_overlap=40, shuffle=True):
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


def construct_path_acs(pheromone, heuristic):
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


def global_pheromone_update(pheromone, path, length):
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


def evaluate_assembly(assembled, original):
    match_len = 0
    for i in range(min(len(assembled), len(original))):
        if assembled[i] == original[i]:
            match_len += 1
        else:
            break
    percent_match = match_len / len(original) * 100
    edit_dist = Levenshtein.distance(assembled, original)
    return percent_match, edit_dist


def run_colony(colony_id, reads, heuristic, shared_best, lock, result_list):
    n = len(reads)
    pheromone = np.full((n, n), TAU0)
    best_path = None
    best_length = float('inf')

    for it in range(NUM_ITER):
        paths = [construct_path_acs(pheromone, heuristic) for _ in range(NUM_ANTS)]
        lengths = [path_length(reads, p) for p in paths]
        idx = np.argmin(lengths)
        current_best_path = paths[idx]
        current_best_length = lengths[idx]

        if current_best_length < best_length:
            best_path = current_best_path
            best_length = current_best_length
            print(f"Colony {colony_id}, Iteration {it}: New best length = {best_length}")

        global_pheromone_update(pheromone, best_path, best_length)

        if it % COOP_INTERVAL == 0:
            with lock:
                if shared_best['length'] is None or best_length < shared_best['length']:
                    shared_best['length'] = best_length
                    shared_best['path'] = best_path
                else:
                    global_pheromone_update(pheromone, shared_best['path'], shared_best['length'])

    result_list[colony_id] = (best_path, best_length)


if __name__ == '__main__':
    start_time = time.time()

    original = random_dna(10000)
    print(original[:200], "\n", original[-200:])
    print("Length original:", len(original))

    reads = generate_reads_from_dna(original, read_length=150, min_overlap=30, max_overlap=120)
    print(f"Generated {len(reads)} reads.")
    print(reads[:5])

    heuristic = build_heuristic(reads)

    manager = Manager()
    shared_best = manager.dict()
    shared_best['path'] = None
    shared_best['length'] = None
    lock = Lock()
    result_list = manager.list([None] * NUM_COLONIES)

    jobs = []
    for i in range(NUM_COLONIES):
        p = multiprocessing.Process(target=run_colony, args=(i, reads, heuristic, shared_best, lock, result_list))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    final_best = min(result_list, key=lambda x: x[1])
    final_path = final_best[0]
    final_sequence = build_sequence(reads, final_path)

    end_time = time.time() - start_time
    percent_match, edit_dist = evaluate_assembly(final_sequence, original)

    print("Original DNA:", original[:100])
    print("Assembled sequence:", final_sequence[:100] + "..." if len(final_sequence) > 100 else final_sequence)
    print(f"Edit distance from original: {edit_dist}")
    print("Original Length:", len(original))
    print("Assembled Length:", len(final_sequence))
    print(f"Total Time:\t {end_time}")
    print(f"Best final length: {final_best[1]}")