# Trabalho 13 - AG com cromossomos reais - Funcao de Rastrigin
# Minimiza f(x, y, z) = 30 + (x^2 - 10*cos(2*pi*x)) + (y^2 - 10*cos(2*pi*y)) + (z^2 - 10*cos(2*pi*z))

import random
import math


DIMENSION = 3
VARIABLE_MIN = -5.12
VARIABLE_MAX = 5.12

POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.20
CROSSOVER_RATE = 0.85
TOURNAMENT_SIZE = 4
ELITE_SIZE = 2

MUTATION_SIGMA = 0.5
CROSSOVER_ALPHA = 0.5


def rastrigin_function(x, y, z):
    """Funcao de Rastrigin 3D para minimizacao."""
    term_x = x * x - 10 * math.cos(2 * math.pi * x)
    term_y = y * y - 10 * math.cos(2 * math.pi * y)
    term_z = z * z - 10 * math.cos(2 * math.pi * z)
    return 30 + term_x + term_y + term_z


def fitness_from_chromosome(chromosome):
    """Calcula aptidao (fitness negativo) a partir de um cromossomo [x, y, z]."""
    x, y, z = chromosome
    return -rastrigin_function(x, y, z)


def create_individual(rng):
    """Cria um individuo aleatorio com cromossomo real em 3D."""
    return [rng.uniform(VARIABLE_MIN, VARIABLE_MAX) for _ in range(DIMENSION)]


def tournament_selection(population, fitness_values, rng, k=TOURNAMENT_SIZE):
    """Seleciona o melhor individuo em um torneio aleatorio."""
    indices = rng.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda i: fitness_values[i])
    return population[best_idx].copy()


def blend_crossover(parent1, parent2, rng, alpha=CROSSOVER_ALPHA):
    """Cruzamento por blend: filho = alpha*pai1 + (1-alpha)*pai2 + ruido."""
    if rng.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    child1 = [alpha * parent1[i] + (1 - alpha) * parent2[i] for i in range(DIMENSION)]
    child2 = [alpha * parent2[i] + (1 - alpha) * parent1[i] for i in range(DIMENSION)]

    for i in range(DIMENSION):
        child1[i] = max(VARIABLE_MIN, min(VARIABLE_MAX, child1[i]))
        child2[i] = max(VARIABLE_MIN, min(VARIABLE_MAX, child2[i]))

    return child1, child2


def gaussian_mutation(chromosome, rng, mutation_rate=MUTATION_RATE, sigma=MUTATION_SIGMA):
    """Mutacao gaussiana: adiciona ruido gaussiano aos genes."""
    mutated = chromosome.copy()

    for i in range(DIMENSION):
        if rng.random() < mutation_rate:
            noise = rng.gauss(0, sigma)
            mutated[i] = mutated[i] + noise
            mutated[i] = max(VARIABLE_MIN, min(VARIABLE_MAX, mutated[i]))

    return mutated


def run_genetic_algorithm(
    population_size=POPULATION_SIZE,
    generations=GENERATIONS,
    mutation_rate=MUTATION_RATE,
    tournament_size=TOURNAMENT_SIZE,
):
    """Executa o AG com cromossomos reais."""
    rng = random.Random(42)
    population = [create_individual(rng) for _ in range(population_size)]

    best_chromosome_ever = None
    best_fitness_ever = float("-inf")

    for generation in range(generations):
        fitness_values = [fitness_from_chromosome(ind) for ind in population]

        current_best_idx = fitness_values.index(max(fitness_values))
        current_best_chromosome = population[current_best_idx].copy()
        current_best_fitness = fitness_values[current_best_idx]

        if current_best_fitness > best_fitness_ever:
            best_fitness_ever = current_best_fitness
            best_chromosome_ever = current_best_chromosome.copy()

        elite = sorted(
            zip(population, fitness_values), key=lambda x: x[1], reverse=True
        )[:ELITE_SIZE]

        new_population = [chromosome.copy() for chromosome, _ in elite]

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_values, rng, k=tournament_size)
            parent2 = tournament_selection(population, fitness_values, rng, k=tournament_size)

            child1, child2 = blend_crossover(parent1, parent2, rng)

            child1 = gaussian_mutation(child1, rng, mutation_rate)
            child2 = gaussian_mutation(child2, rng, mutation_rate)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population[:population_size]

        if (generation + 1) % 50 == 0 or generation == 0:
            best_value = -best_fitness_ever
            print(
                f"Geração {generation + 1:3d} | "
                f"Melhor f(x,y,z): {best_value:.6f}"
            )

    return best_chromosome_ever, best_fitness_ever


if __name__ == "__main__":
    print("=" * 60)
    print("ALGORITMO GENETICO COM CROMOSSOMOS REAIS")
    print("Funcao de Rastrigin 3D")
    print("=" * 60)
    print(f"Dominio: [{VARIABLE_MIN}, {VARIABLE_MAX}]³")
    print(f"Populacao: {POPULATION_SIZE}")
    print(f"Geracoes: {GENERATIONS}")
    print(f"Taxa de mutacao: {MUTATION_RATE}")
    print(f"Taxa de cruzamento: {CROSSOVER_RATE}")
    print(f"Desvio padrao (sigma): {MUTATION_SIGMA}")
    print("=" * 60 + "\n")

    best_solution, best_fitness = run_genetic_algorithm()
    best_value = -best_fitness

    x_opt, y_opt, z_opt = best_solution

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print(f"Melhor cromossomo: [{x_opt:.6f}, {y_opt:.6f}, {z_opt:.6f}]")
    print(f"Valor otimizado f(x,y,z): {best_value:.6f}")
    print(f"(Minimo global esperado: f(0,0,0) = 0)")
    print("=" * 60)
