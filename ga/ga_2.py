# Trabalho 09 - AG Parametrizável 
# Encontra o mínimo de f(x) = -|x*sin(sqrt(|x|))| no intervalo [0, 512]

import random
import math

def binary_to_decimal(binary_string):
    decimal = int(binary_string, 2)
    return (decimal / (2**len(binary_string) - 1)) * 512.0

def objective_function(x):
    return -abs(x * math.sin(math.sqrt(abs(x))))

def fitness(chromosome):
    x = binary_to_decimal(chromosome)
    return objective_function(x)

def create_random_chromosome(bits):
    return ''.join(str(random.randint(0, 1)) for _ in range(bits))

def create_initial_population(population_size, chromosome_bits):
    return [create_random_chromosome(chromosome_bits) for _ in range(population_size)]

def selection_roulette(population):
    fitness_values = [fitness(chromosome) for chromosome in population]
    
    min_fitness = min(fitness_values)
    max_fitness = max(fitness_values)
    
    if max_fitness == min_fitness:
        probabilities = [1 / len(population)] * len(population)
    else:
        inverted_fitness = [max_fitness - f for f in fitness_values]
        total_fitness = sum(inverted_fitness)
        probabilities = [f / total_fitness for f in inverted_fitness]
    
    parent1 = random.choices(population, weights=probabilities, k=1)[0]
    parent2 = random.choices(population, weights=probabilities, k=1)[0]
    
    return parent1, parent2

def selection_tournament(population, tournament_size):
    def tournament():
        candidates = random.sample(population, min(tournament_size, len(population)))
        return min(candidates, key=fitness)
    
    parent1 = tournament()
    parent2 = tournament()
    
    return parent1, parent2

def crossover_one_point(parent1, parent2, crossover_prob):
    if random.random() > crossover_prob:
        return parent1, parent2
    
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    
    return child1, child2

def crossover_two_point(parent1, parent2, crossover_prob):
    if random.random() > crossover_prob:
        return parent1, parent2
    
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)
    
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    
    return child1, child2

def mutate(chromosome, mutation_prob):
    if random.random() > mutation_prob:
        return chromosome
    
    chromosome_list = list(chromosome)
    mutation_point = random.randint(0, len(chromosome) - 1)
    chromosome_list[mutation_point] = '0' if chromosome_list[mutation_point] == '1' else '1'
    
    return ''.join(chromosome_list)

def get_parameters():
    print("\n" + "=" * 60)
    print("ALGORITMO GENÉTICO PARAMETRIZÁVEL")
    print("=" * 60 + "\n")
    
    population_size = int(input("Tamanho da população: "))
    chromosome_bits = int(input("Bits por cromossomo: "))
    max_generations = int(input("Quantidade máxima de gerações: "))
    crossover_percentage = float(input("Porcentagem de cruzamento (0-1): "))
    mutation_probability = float(input("Probabilidade de mutação (0-1): "))
    
    print("\nMétodos de seleção:")
    print("1. Roleta")
    print("2. Torneio")
    selection_method = int(input("Escolha o método (1 ou 2): "))
    
    tournament_size = None
    if selection_method == 2:
        tournament_size = int(input("Tamanho do torneio: "))
    
    print("\nMétodos de cruzamento:")
    print("1. Um ponto")
    print("2. Dois pontos")
    crossover_method = int(input("Escolha o método (1 ou 2): "))
    
    return {
        'population_size': population_size,
        'chromosome_bits': chromosome_bits,
        'max_generations': max_generations,
        'crossover_percentage': crossover_percentage,
        'mutation_probability': mutation_probability,
        'selection_method': selection_method,
        'tournament_size': tournament_size,
        'crossover_method': crossover_method
    }

def genetic_algorithm(params):
    
    print("\n" + "=" * 60)
    print("INICIANDO ALGORITMO GENÉTICO")
    print("=" * 60)
    print(f"Objetivo: minimizar f(x) = -|x*sin(sqrt(|x|))| em [0, 512]")
    print(f"Tamanho da população: {params['population_size']}")
    print(f"Bits por cromossomo: {params['chromosome_bits']}")
    print(f"Gerações: {params['max_generations']}")
    print(f"Mutação: {params['mutation_probability']}")
    print(f"Crossover: {params['crossover_percentage']}")
    selection_name = "Roleta" if params['selection_method'] == 1 else "Torneio"
    print(f"Seleção: {selection_name}")
    crossover_name = "Um ponto" if params['crossover_method'] == 1 else "Dois pontos"
    print(f"Cruzamento: {crossover_name}")
    print("=" * 60 + "\n")
    
    population = create_initial_population(params['population_size'], params['chromosome_bits'])
    best_chromosome = min(population, key=fitness)
    best_fitness = fitness(best_chromosome)
    
    for generation in range(params['max_generations']):
        new_population = []
        
        while len(new_population) < params['population_size']:
            if params['selection_method'] == 1:
                parent1, parent2 = selection_roulette(population)
            else:
                parent1, parent2 = selection_tournament(population, params['tournament_size'])
            
            if params['crossover_method'] == 1:
                child1, child2 = crossover_one_point(parent1, parent2, params['crossover_percentage'])
            else:
                child1, child2 = crossover_two_point(parent1, parent2, params['crossover_percentage'])
            
            child1 = mutate(child1, params['mutation_probability'])
            child2 = mutate(child2, params['mutation_probability'])
            
            new_population.append(child1)
            if len(new_population) < params['population_size']:
                new_population.append(child2)
        
        population = new_population[:params['population_size']]
        
        current_best = min(population, key=fitness)
        current_best_fitness = fitness(current_best)
        
        if current_best_fitness < best_fitness:
            best_chromosome = current_best
            best_fitness = current_best_fitness
        
        if (generation + 1) % 10 == 0 or generation == 0:
            x_best = binary_to_decimal(best_chromosome)
            print(f"Geração {generation + 1:3d} | Melhor x: {x_best:7.3f} | Melhor f(x): {best_fitness:9.6f}")
    
    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS")
    print("=" * 60)
    
    x_best = binary_to_decimal(best_chromosome)
    print(f"Melhor cromossomo (binário): {best_chromosome}")
    print(f"Melhor valor de x: {x_best:.6f}")
    print(f"Melhor valor de f(x): {best_fitness:.6f}")
    print(f"Mínimo encontrado: {best_fitness:.6f}")
    
    return best_chromosome, x_best, best_fitness

if __name__ == "__main__":
    random.seed(42)
    params = get_parameters()
    best_chromosome, x_best, best_fitness = genetic_algorithm(params)
