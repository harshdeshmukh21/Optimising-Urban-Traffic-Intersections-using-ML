import random
import csv

data = []
with open('imps.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        data.append(row)

SWARM_SIZE = 100
ITERATIONS = 200
MIN_CYCLE = 60
MAX_CYCLE = 120
C1_START, C1_END = 2.0, 0.5  
C2_START, C2_END = 0.5, 2.0  
W_START, W_END = 0.9, 0.4  

SATURATION_FLOW = 1800  
QUEUE_CAPACITY = 600  
MIN_GREEN_TIME = 10  

class Particle:
    def __init__(self):
        self.position = {
            'C': random.randint(MIN_CYCLE, MAX_CYCLE),
            'g1_ratio': random.uniform(0.4, 0.6)  
        }
        self.velocity = {
            'C': random.uniform(-5, 5), 
            'g1_ratio': random.uniform(-0.05, 0.05)
        }
        self.best_position = self.position.copy()
        self.best_score = float('-inf')

def calculate_green_times(position):
    total_green = position['C'] - 12  
    g1 = max(MIN_GREEN_TIME, int(total_green * position['g1_ratio']))
    g2 = max(MIN_GREEN_TIME, total_green - g1)
    return g1, g2

def calculate_ett(C, g1, g2, q1=600, q2=400):
    g1 = max(MIN_GREEN_TIME, g1)
    g2 = max(MIN_GREEN_TIME, g2)
    
    d1 = (q1 * g1) / (SATURATION_FLOW - q1) if (SATURATION_FLOW - q1) > 0 else float('inf')
    d2 = (q2 * g2) / (SATURATION_FLOW - q2) if (SATURATION_FLOW - q2) > 0 else float('inf')
    
    avg_delay = (d1 + d2) / 2
    
    ett = C + (avg_delay * 3600)  
    return max(ett, C)

def fitness(position, traffic_data):
    g1, g2 = calculate_green_times(position)
    highway_demand = max(1, float(traffic_data['Traffic demands from highway (B+F)']))
    bridge_demand = max(1, float(traffic_data['Traffic demands from bridge (A+E)']))
    
    C = position['C']
    ett = calculate_ett(C, g1, g2, q1=highway_demand, q2=bridge_demand)
    
    cycle_efficiency = 1 - abs(C - 90) / 60 
    ett_score = 1 / ett  
    
    return cycle_efficiency * ett_score

def update_velocity(particle, global_best_position, w, c1, c2):
    for key in particle.velocity.keys():
        r1, r2 = random.random(), random.random()
        cognitive = c1 * r1 * (particle.best_position[key] - particle.position[key])
        social = c2 * r2 * (global_best_position[key] - particle.position[key])
        particle.velocity[key] = w * particle.velocity[key] + cognitive + social

def update_position(particle):
    for key in particle.position.keys():
        particle.position[key] += particle.velocity[key]
        if key == 'C':
            particle.position[key] = max(MIN_CYCLE, min(MAX_CYCLE, particle.position[key]))
        else:
            particle.position[key] = max(0.4, min(0.6, particle.position[key]))

def local_search(position):
    new_position = position.copy()
    key = random.choice(list(position.keys()))
    if key == 'C':
        new_position[key] += random.randint(-5, 5)  # Larger perturbation for cycle length
        new_position[key] = max(MIN_CYCLE, min(MAX_CYCLE, new_position[key]))
    else:
        new_position[key] += random.uniform(-0.03, 0.03)  # Slightly larger variation
        new_position[key] = max(0.4, min(0.6, new_position[key]))
    return new_position

def particle_swarm_optimization(traffic_data_list):
    swarm = [Particle() for _ in range(SWARM_SIZE)]
    global_best_position = None
    global_best_score = float('-inf')
    best_solutions = []
    stagnation_counter = 0
    last_best_score = global_best_score

    for iteration in range(ITERATIONS):
        w = W_START - (W_START - W_END) * (iteration / ITERATIONS)
        c1 = C1_START - (C1_START - C1_END) * (iteration / ITERATIONS)
        c2 = C2_START + (C2_END - C2_START) * (iteration / ITERATIONS)
        
        for particle in swarm:
            score = sum(fitness(particle.position, td) for td in traffic_data_list)
            
            if score > particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
            
            if score > global_best_score:
                global_best_score = score
                global_best_position = particle.position.copy()

        for particle in swarm:
            update_velocity(particle, global_best_position, w, c1, c2)
            update_position(particle)
            
            if random.random() < 0.2:  # Increased local search probability
                new_position = local_search(particle.position)
                new_score = sum(fitness(new_position, td) for td in traffic_data_list)
                if new_score > particle.best_score:
                    particle.position = new_position
                    particle.best_position = new_position
                    particle.best_score = new_score

        if abs(global_best_score - last_best_score) < 1e-5:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        last_best_score = global_best_score

        if stagnation_counter > 10:  # Reset some particles to avoid stagnation
            for particle in swarm:
                if random.random() < 0.3:
                    particle.position = {
                        'C': random.randint(MIN_CYCLE, MAX_CYCLE),
                        'g1_ratio': random.uniform(0.4, 0.6)
                    }
            stagnation_counter = 0

        g1, g2 = calculate_green_times(global_best_position)
        C = round(global_best_position['C'])
        ett = calculate_ett(C, g1, g2, 
                            q1=float(traffic_data_list[0]['Traffic demands from highway (B+F)']),
                            q2=float(traffic_data_list[0]['Traffic demands from bridge (A+E)']))
        best_solutions.append((C, g1, g2, round(ett, 2)))

    # Saving best solutions with ETT to a CSV file
    with open("pso_best_solutions_with_ett.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["C", "g1", "g2", "ett"])
        writer.writerows(best_solutions)

    return best_solutions

# Running PSO and generating the CSV
print("\nRunning PSO algorithm for all rows")
best_solutions = particle_swarm_optimization(data)
