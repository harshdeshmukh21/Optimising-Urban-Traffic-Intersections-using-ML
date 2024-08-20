import random
import csv

# Read data from CSV file
data = []
with open('imps.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        data.append(row)

POPULATION_SIZE = 20
GENERATIONS = 200
MUTATION_RATE = 0.1
MIN_CYCLE = 60
MAX_CYCLE = 120

def initialize_population():
    return [
        {
            'C': random.randint(MIN_CYCLE, MAX_CYCLE),
            'g1': random.uniform(0.1, 0.9),
            'g2': random.uniform(0.1, 0.9)
        }
        for _ in range(POPULATION_SIZE)
    ]

def calculate_green_times(solution):
    total_green = solution['C'] - 12
    g1 = max(1, int(total_green * solution['g1'] / (solution['g1'] + solution['g2'])))
    g2 = max(1, total_green - g1)
    return g1, g2

def fitness(solution, traffic_data):
    g1, g2 = calculate_green_times(solution)
    highway_demand = max(1, float(traffic_data['Traffic demands from highway (B+F)']))
    bridge_demand = max(1, float(traffic_data['Traffic demands from bridge (A+E)']))
    green_ratio = g1 / g2
    demand_ratio = highway_demand / bridge_demand
    epsilon = 1e-6
    return 1 / (abs(green_ratio - demand_ratio) + epsilon)

def select_parents(population, traffic_data):
    return sorted(population, key=lambda x: fitness(x, traffic_data), reverse=True)[:POPULATION_SIZE//2]

def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(solution):
    if random.random() < MUTATION_RATE:
        key = random.choice(['C', 'g1', 'g2'])
        if key == 'C':
            solution[key] = random.randint(MIN_CYCLE, MAX_CYCLE)
        else:
            solution[key] = random.uniform(0.1, 0.9)
    return solution

def genetic_algorithm(traffic_data_list):
    population = initialize_population()
    all_solutions = []
    for generation in range(GENERATIONS):
        offspring = []
        for traffic_data in traffic_data_list:
            parents = select_parents(population, traffic_data)
            while len(offspring) < POPULATION_SIZE:
                parent1, parent2 = random.sample(parents, 2)
                child = crossover(parent1, parent2)
                child = mutate(child)
                offspring.append(child)
        population = offspring
        best_solution = max(population, key=lambda x: sum(fitness(x, td) for td in traffic_data_list))
        g1, g2 = calculate_green_times(best_solution)
        all_solutions.append((best_solution['C'], g1, g2))
    return all_solutions

print("\nRunning genetic algorithm for all rows")
all_solutions = genetic_algorithm(data)

# Write results to CSV file
output_filename = 'genetic_algorithm_results.csv'
with open(output_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write header
    csvwriter.writerow(['cycle_length', 'g1', 'g2'])
    # Write data
    for solution in all_solutions:
        csvwriter.writerow(solution)

print(f"\nResults have been written to {output_filename}")

# Print best solutions throughout generations (optional)
print("\nBest solutions throughout generations:")
for gen, solution in enumerate(all_solutions):
    print(f"Generation {gen + 1}: C={solution[0]}, g1={solution[1]}, g2={solution[2]}")

# Find overall best solution
best_solution = max(all_solutions, key=lambda x: x[1] + x[2])  # Maximize total green time

print(f"\nOverall best solution for all rows:")
print(f"C={best_solution[0]}, g1={best_solution[1]}, g2={best_solution[2]}")