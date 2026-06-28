# Trabalho 14 - Caixeiro viajante com Ranking Linear
# Resolve o problema do caixeiro viajante com AG usando selecao por Ranking Linear

import random
import pandas as pd
from pathlib import Path

CITIES_FILE = "data/cities.xlsx"
UNREACHABLE_DISTANCE = 999999

# Parametros de Ranking Linear
RANKING_MIN = 0.5
RANKING_MAX = 2.5


def load_distance_dataframe():
    local_path = Path(__file__).resolve().parent / CITIES_FILE
    if local_path.exists():
        return pd.read_excel(local_path, index_col=0)

    # Backward compatible fallback if the user keeps the file in /data.
    return pd.read_excel("/data/cities.xlsx", index_col=0)


distance_df = load_distance_dataframe()

cities = list(distance_df.index)
NUM_CITIES = len(cities)

for city in cities:
    distance_df.loc[city, city] = 0

initial_matrix = distance_df.fillna(UNREACHABLE_DISTANCE).to_numpy().astype(float)

distance_matrix = initial_matrix.copy()
for k in range(NUM_CITIES):
    for i in range(NUM_CITIES):
        for j in range(NUM_CITIES):
            if distance_matrix[i][k] + distance_matrix[k][j] < distance_matrix[i][j]:
                distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]


def calculate_route_distance(route):
    total_distance = 0
    for i in range(NUM_CITIES - 1):
        origin = route[i]
        destination = route[i + 1]
        total_distance += distance_matrix[origin][destination]
    return total_distance


def create_individual():
    route = list(range(NUM_CITIES))
    random.shuffle(route)
    return route


def linear_ranking_selection(population, population_distances, rank_min=RANKING_MIN, rank_max=RANKING_MAX):
    """
    Selecao por Ranking Linear.
    Ordena a populacao por fitness e atribui probabilidades lineares aos ranks.
    Melhor fitness recebe rank_max, pior recebe rank_min.
    """
    n = len(population)
    
    # Ordena indices por distancia (menor distancia = melhor fitness)
    ranked_indices = sorted(range(n), key=lambda i: population_distances[i])
    
    # Calcula a probabilidade para cada rank
    probabilities = []
    for rank in range(n):
        prob = rank_min + (rank_max - rank_min) * rank / (n - 1)
        probabilities.append(prob)
    
    # Cria mapa: indice original -> probabilidade
    prob_map = [0.0] * n
    for i, original_idx in enumerate(ranked_indices):
        prob_map[original_idx] = probabilities[i]
    
    # Normaliza probabilidades
    total_prob = sum(prob_map)
    normalized_probs = [p / total_prob for p in prob_map]
    
    # Seleciona dois individuos usando as probabilidades
    selected_indices = random.choices(range(n), weights=normalized_probs, k=2)
    return population[selected_indices[0]].copy(), population[selected_indices[1]].copy()


def ordered_crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(NUM_CITIES), 2))
    child = [None] * NUM_CITIES
    child[point1 : point2 + 1] = parent1[point1 : point2 + 1]

    child_pos = (point2 + 1) % NUM_CITIES
    parent2_pos = (point2 + 1) % NUM_CITIES

    while None in child:
        gene = parent2[parent2_pos]
        if gene not in child:
            child[child_pos] = gene
            child_pos = (child_pos + 1) % NUM_CITIES
        parent2_pos = (parent2_pos + 1) % NUM_CITIES

    return child


def swap_mutation(route, mutation_rate=0.05):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(NUM_CITIES), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route


def run_genetic_algorithm(population_size=200, generations=500, mutation_rate=0.20):
    population = [create_individual() for _ in range(population_size)]

    best_distance_ever = float("inf")
    best_route_ever = []

    for generation in range(generations):
        distances = [calculate_route_distance(individual) for individual in population]

        min_distance = min(distances)
        if min_distance < best_distance_ever:
            best_distance_ever = min_distance
            best_route_ever = population[distances.index(min_distance)].copy()

        new_population = []
        new_population.append(best_route_ever)

        while len(new_population) < population_size:
            parent1, parent2 = linear_ranking_selection(population, distances)

            child = ordered_crossover(parent1, parent2)
            child = swap_mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population

        if (generation + 1) % 100 == 0 or generation == 0:
            print(f"Geração {generation+1:03d} | Melhor distância linear: {best_distance_ever:.0f} km")

    return best_route_ever, best_distance_ever


if __name__ == "__main__":
    print("=" * 60)
    print("ALGORITMO GENETICO - CAIXEIRO VIAJANTE")
    print("Selecao por Ranking Linear")
    print("=" * 60)
    print(f"Ranking Min: {RANKING_MIN}, Ranking Max: {RANKING_MAX}")
    print("=" * 60 + "\n")

    best_route_indices, best_distance = run_genetic_algorithm()
    route_names = [cities[idx] for idx in best_route_indices]

    print("\n" + "=" * 60)
    print("RESULTADO FINAL (DISTÂNCIA REAL)")
    print("=" * 60)
    print(f"Distância Total Otimizada: {best_distance:.0f} km")
    print("\nSequência de Cidades:")
    print(" -> ".join(route_names))
    print("=" * 60)
