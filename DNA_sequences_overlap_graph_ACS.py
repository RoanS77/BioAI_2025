import random
import numpy as np
import time

# ACS params
ALPHA = 1.0
BETA = 2.0
RHO = 0.1
PHI = 0.1
Q0 = 0.9
Q = 100
NUM_ANTS = 20
NUM_ITER = 20
TAU0 = 0.1

def overlap(a, b, min_length=20):
    max_len = min(len(a), len(b))
    for i in range(max_len, min_length - 1, -1):
        if a[-i:] == b[:i]:
            return i
    return 0

def random_dna(length):
    return ''.join(random.choices('ACGT', k=length))

def generate_repetitive_dna(total_length=5000, repeat_unit_length=20, mutation_rate=0.01):
    bases = ['A', 'C', 'G', 'T']
    repeat_unit = ''.join(random.choices(bases, k=repeat_unit_length))
    repeats = total_length // repeat_unit_length

    sequence = []
    for _ in range(repeats):
        unit = list(repeat_unit)
        for i in range(len(unit)):
            if random.random() < mutation_rate:
                unit[i] = random.choice([b for b in bases if b != unit[i]])
        sequence.append(''.join(unit))

    return ''.join(sequence)

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

# generate test data
original = random_dna(10000)
# original = random_dna(200) + generate_repetitive_dna(5000, 12, 0.001) + random_dna(200)
print(original[:200], "\n", original[-200:])
print("Length original:", len(original))
reads = generate_reads_from_dna(original, read_length=150, min_overlap=30, max_overlap=120)
print(f"Generated {len(reads)} reads.")
print(reads[:5])

n = len(reads)
heuristic = np.zeros((n, n))
pheromone = np.full((n, n), TAU0)

for i in range(n):
    for j in range(n):
        if i != j:
            heuristic[i][j] = overlap(reads[i], reads[j])

max_heuristic = np.max(heuristic)
if max_heuristic > 0:
    heuristic = heuristic / max_heuristic

def local_pheromone_update(current, next_node):
    global pheromone
    pheromone[current][next_node] = (1 - PHI) * pheromone[current][next_node] + PHI * TAU0

def construct_path_acs():
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

        if q <= Q0:  # exploitation
            best_j = None
            best_value = -1

            for j in candidates:
                tau = pheromone[current][j]
                eta = heuristic[current][j]
                value = tau * (eta ** BETA)

                if value > best_value:
                    best_value = value
                    best_j = j

            next_node = best_j if best_j is not None else random.choice(candidates)

        else:  # exploration
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

        local_pheromone_update(current, next_node)
        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return path

def global_pheromone_update(best_path, best_length):
    global pheromone
    pheromone *= (1 - RHO)

    if best_path and len(best_path) > 1:
        pheromone_deposit = Q / best_length
        for i in range(len(best_path) - 1):
            a, b = best_path[i], best_path[i + 1]
            pheromone[a][b] += RHO * pheromone_deposit

def path_length(path):
    total = len(reads[path[0]])
    for i in range(1, len(path)):
        olap = overlap(reads[path[i - 1]], reads[path[i]])
        total += len(reads[path[i]]) - olap
    return total

def build_sequence(path):
    sequence = reads[path[0]]
    for i in range(1, len(path)):
        olap = overlap(sequence, reads[path[i]])
        sequence += reads[path[i]][olap:]
    return sequence

# main loop
start_time = time.time()
best_path = None
best_length = float('inf')

for it in range(NUM_ITER):
    all_paths = [construct_path_acs() for _ in range(NUM_ANTS)]
    all_lengths = [path_length(p) for p in all_paths]

    iteration_best_idx = np.argmin(all_lengths)
    iteration_best_length = all_lengths[iteration_best_idx]
    iteration_best_path = all_paths[iteration_best_idx]

    if iteration_best_length < best_length:
        best_length = iteration_best_length
        best_path = iteration_best_path
        print(f"Iteration {it}: New best length = {best_length}")

    global_pheromone_update(best_path, best_length)

end_time = time.time() - start_time

# visualization and results
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import Levenshtein

def visualize_path(reads, path):
    G = nx.DiGraph()
    overlaps = []

    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        olap = overlap(reads[a], reads[b])
        overlaps.append(olap)
        G.add_edge(f"R{a}", f"R{b}", label=str(olap), weight=olap)

    pos = nx.kamada_kawai_layout(G)

    node_colors = []
    for i, node in enumerate(path):
        if i == 0:
            node_colors.append("limegreen")
        elif i == len(path) - 1:
            node_colors.append("crimson")
        else:
            node_colors.append("skyblue")

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    if weights:
        norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
        edge_cmap = cm.get_cmap("Blues")
        edge_colors = [edge_cmap(norm(w)) for w in weights]
        edge_widths = [1 + 2 * norm(w) for w in weights]
    else:
        edge_colors = "blue"
        edge_widths = 1

    plt.figure(figsize=(10, 7))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=100,
        node_color=node_colors,
        font_size=7,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=20,
        connectionstyle='arc3,rad=0.05'
    )

    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

    plt.title("ACS Best Path Through Reads", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

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

assembled = build_sequence(best_path)
percent_match, edit_dist = evaluate_assembly(assembled, original)

print("Original DNA:", original[:100])
print("Assembled sequence:", assembled[:100] + "..." if len(assembled) > 100 else assembled)
print(f"Edit distance from original: {edit_dist}")
print("Original Length:", len(original))
print("Assembled Length:", len(assembled))
print(f"Total Time:\t {end_time}")

visualize_path(reads, best_path)