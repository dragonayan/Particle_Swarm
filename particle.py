import numpy as np
import random
import time
import json

# Hexagonal grid directions (even-r coordinate system)
DIRECTIONS = [
    (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)
]

class Particle:
    def __init__(self, position, velocity, personal_best):
        self.position = position
        self.velocity = velocity
        self.personal_best = personal_best

def hex_distance(a, b):
    return (abs(a[0] - b[0]) + abs(a[0] + a[1] - b[0] - b[1]) + abs(a[1] - b[1])) // 2

def get_neighbors(pos, n, m, obstacles):
    neighbors = []
    for d in DIRECTIONS:
        new_pos = (pos[0] + d[0], pos[1] + d[1])
        if 0 <= new_pos[0] < n and 0 <= new_pos[1] < m and new_pos not in obstacles:
            neighbors.append(new_pos)
    return neighbors

def pso_hexagonal_routing(n, m, obstacles, start, end, max_iter=1000, num_particles=100):
    particles = []
    for _ in range(num_particles):
        position = start
        velocity = random.choice(DIRECTIONS)
        personal_best = position
        particles.append(Particle(position, velocity, personal_best))

    global_best = start
    global_best_distance = hex_distance(start, end)

    for iter in range(max_iter):
        for particle in particles:
            # Update particle velocity and position
            new_velocity = random.choice(DIRECTIONS)
            new_position = (particle.position[0] + new_velocity[0], particle.position[1] + new_velocity[1])
            if 0 <= new_position[0] < n and 0 <= new_position[1] < m and new_position not in obstacles:
                particle.velocity = new_velocity
                particle.position = new_position
            
            # Introduce additional computationally intensive operation
            dummy_operation = sum([i**2 for i in range(1000)])
            
            # Update personal best
            if hex_distance(particle.position, end) < hex_distance(particle.personal_best, end):
                particle.personal_best = particle.position
            
            # Update global best
            if hex_distance(particle.personal_best, end) < global_best_distance:
                global_best = particle.personal_best
                global_best_distance = hex_distance(global_best, end)

        if global_best == end:
            break
    
    path = []
    current_pos = start
    while current_pos != end:
        path.append(current_pos)
        neighbors = get_neighbors(current_pos, n, m, obstacles)
        current_pos = min(neighbors, key=lambda pos: hex_distance(pos, end))
    path.append(end)
    return path

def save_test_cases(test_cases, filename):
    with open(filename, "w") as f:
        json.dump(test_cases, f, indent=4)

def save_results(results, filename):
    with open(filename, "w") as f:
        for result in results:
            f.write(f"Grid Size: {result['grid_size']},  Execution Time: {result['execution_time']:.4f} seconds\n")

def test_pso_routing(test_cases):
    results = []
    for i, test in enumerate(test_cases):
        n = test["n"]
        m = test["m"]
        obstacles = set(map(tuple, test["obstacles"]))
        start = tuple(test["start"])
        end = tuple(test["end"])

        start_time = time.time()
        path = pso_hexagonal_routing(n, m, obstacles, start, end)
        end_time = time.time()
        execution_time = end_time - start_time

        result = {
            "grid_size": (n, m),
            "path": path,
            "execution_time": execution_time
        }
        results.append(result)
        print(f"Test Case {i + 1}: Grid size: {n}x{m}, Path: {path}, Execution Time: {execution_time:.4f} seconds")

        # Save individual result to file
        with open(f"result_{i + 1}.txt", "w") as f:
            f.write(f"Grid Size: {result['grid_size']},  Execution Time: {result['execution_time']:.4f} seconds\n")

    # Save the dataset of grid size vs required time
    save_results(results, "execution_times.txt")

    return results

# Define new test cases
new_test_cases = [
    {
		"n": 10,
		"m": 10,
		"obstacles": [
			[2, 1],
			[2, 2],
			[2, 3],
			[3, 4],
			[4, 4],
			[4, 6],
			[4, 8],
			[5, 2],
			[5, 6],
			[5, 8],
			[6, 3],
			[6, 5],
			[7, 8],
			[8, 1],
			[8, 3],
			[8, 5]
		],
		"start": [8, 7],
		"end": [3, 2]
	},
    {
		"n": 15,
		"m": 15,
		"obstacles": [
			[2, 2],
			[2, 4],
			[2, 5],
			[2, 7],
			[2, 9],
			[3, 12],
			[4, 9],
			[4, 10],
			[5, 0],
			[5, 5],
			[5, 7],
			[5, 13],
			[6, 2],
			[6, 5],
			[7, 2],
			[7, 4],
			[7, 10],
			[7, 11],
			[7, 14],
			[8, 1],
			[8, 3],
			[8, 6],
			[8, 7],
			[8, 11],
			[8, 14],
			[9, 3],
			[9, 4],
			[9, 10],
			[10, 6],
			[10, 8],
			[10, 10],
			[10, 11],
			[11, 1],
			[12, 3],
			[12, 6],
			[12, 9],
			[12, 10],
			[13, 7]
		],
		"start": [7, 7],
		"end": [12, 2]
	},
    {
        "n": 20,
        "m": 20,
        "obstacles": [
            [3, 4], [3, 10], [3, 12], [3, 16], [3, 17],
            [4, 1], [4, 2], [4, 4], [4, 7], [5, 5],
            [5, 11], [5, 13], [5, 14], [6, 4], [6, 10],
            [6, 15], [7, 4], [7, 7], [7, 9], [7, 12],
            [7, 15], [8, 4], [8, 12], [9, 6], [10, 2],
            [10, 5], [10, 6], [10, 13], [11, 2], [11, 5],
            [11, 13], [12, 4], [12, 7], [12, 9], [12, 11],
            [12, 13], [12, 16], [13, 2], [13, 12], [13, 17],
            [14, 3], [14, 6], [14, 8], [14, 15], [15, 3],
            [15, 12], [16, 6], [16, 7], [16, 12], [17, 17]
        ],
        "start": [2, 2],
        "end": [18, 14]
    },
    {
        "n": 25,
        "m": 25,
        "obstacles": [
            [0, 11], [1, 10], [2, 12], [3, 5], [3, 8], 
            [3, 9], [3, 10], [3, 14], [3, 15], [4, 10],
            [4, 12], [4, 17], [6, 11], [6, 13], [6, 14], 
            [7, 11], [7, 13], [8, 11], [9, 10], [11, 11], 
            [11, 13], [12, 11], [13, 10], [13, 13], [14, 11], 
            [15, 11], [15, 12], [15, 13], [16, 6], [17, 7], 
            [17, 12], [17, 14], [18, 11], [18, 14], [19, 11], 
            [19, 13], [20, 10], [20, 18], [21, 7], [21, 11], 
            [22, 11], [22, 13], [22, 14], [24, 9]
        ],
        "start": [10, 4],
        "end": [15, 20]
    },
    {
        "n": 30,
        "m": 30,
        "obstacles": [
            [4, 7], [4, 9], [5, 11], [5, 23], [6, 7], 
            [6, 8], [6, 9], [6, 14], [6, 17], [6, 18], 
            [6, 19], [7, 6], [7, 20], [8, 4], [8, 10], 
            [9, 9], [9, 11], [9, 14], [9, 15], [9, 24], 
            [10, 4], [10, 9], [10, 18], [10, 20], [10, 23], 
            [11, 3], [11, 14], [11, 19], [11, 20], [11, 24], 
            [12, 7], [12, 14], [12, 23], [13, 4], [13, 15], 
            [13, 19], [13, 23], [14, 4], [14, 8], [14, 9], 
            [14, 14], [14, 15], [14, 16], [14, 17], [14, 19], 
            [14, 22], [15, 10], [15, 15], [16, 4], [16, 6], 
                        [16, 9], [16, 12], [16, 18], [16, 20], [16, 23],
            [17, 2], [17, 5], [17, 14], [17, 19], [17, 23],
            [18, 4], [18, 6], [18, 7], [18, 8], [18, 11],
            [18, 19], [19, 4], [19, 6], [19, 9], [19, 16],
            [20, 7], [20, 12], [20, 17], [20, 21], [20, 23],
            [21, 7], [21, 9], [21, 12], [21, 19], [22, 5],
            [22, 8], [22, 14], [22, 15], [22, 16], [22, 20],
            [23, 7], [23, 8], [23, 10], [23, 16], [23, 21],
            [23, 23], [24, 4], [24, 6], [24, 12], [24, 15],
            [24, 17], [24, 23], [25, 4], [25, 7], [25, 9],
            [25, 13], [25, 15], [25, 18], [25, 23], [26, 2],
            [26, 3], [26, 4], [26, 6], [26, 7], [26, 8],
            [26, 10], [26, 16], [26, 17], [26, 18], [26, 23],
            [27, 4], [27, 6], [27, 9], [27, 12], [27, 18],
            [27, 21], [27, 23], [28, 4], [28, 9], [28, 14],
            [28, 19], [28, 23], [29, 7], [29, 13], [29, 17],
            [29, 19], [29, 23]
        ],
        "start": [5, 5],
        "end": [25, 25]
    },
    {
		"n": 35,
		"m": 35,
		"obstacles": [
			[3, 24],
			[3, 26],
			[3, 27],
			[4, 10],
			[4, 16],
			[4, 29],
			[5, 20],
			[6, 7],
			[6, 9],
			[6, 13],
			[6, 19],
			[6, 24],
			[6, 28],
			[7, 13],
			[7, 16],
			[7, 18],
			[7, 21],
			[7, 24],
			[7, 27],
			[8, 19],
			[8, 26],
			[9, 9],
			[9, 16],
			[9, 19],
			[9, 20],
			[10, 7],
			[10, 9],
			[10, 11],
			[10, 14],
			[10, 23],
			[10, 28],
			[11, 6],
			[11, 8],
			[11, 17],
			[11, 25],
			[11, 26],
			[11, 29],
			[12, 4],
			[12, 5],
			[12, 6],
			[12, 8],
			[12, 13],
			[12, 18],
			[12, 26],
			[12, 29],
			[13, 7],
			[13, 12],
			[13, 24],
			[13, 27],
			[14, 18],
			[14, 19],
			[14, 25],
			[14, 27],
			[15, 7],
			[15, 10],
			[15, 15],
			[15, 19],
			[15, 27],
			[16, 4],
			[16, 8],
			[16, 11],
			[16, 18],
			[16, 20],
			[17, 4],
			[17, 18],
			[17, 22],
			[17, 23],
			[17, 28],
			[18, 4],
			[19, 12],
			[19, 14],
			[19, 18],
			[19, 20],
			[19, 26],
			[20, 4],
			[20, 7],
			[20, 9],
			[20, 24],
			[20, 26],
			[20, 31],
			[21, 4],
			[21, 9],
			[21, 18],
			[21, 27],
			[21, 28],
			[22, 12],
			[22, 16],
			[22, 23],
			[22, 28],
			[23, 4],
			[23, 13],
			[23, 21],
			[23, 26],
			[23, 28],
			[23, 29],
			[24, 11],
			[24, 17],
			[24, 27],
			[25, 4],
			[25, 7],
			[25, 8],
			[25, 14],
			[26, 13],
			[26, 18],
			[26, 19],
			[26, 20],
			[26, 22],
			[26, 23],
			[27, 10],
			[27, 21],
			[27, 28],
			[28, 10],
			[28, 12],
			[28, 13],
			[28, 17],
			[29, 10],
			[29, 17],
			[29, 18],
			[29, 28],
			[30, 10]
		],
		"start": [4, 22],
		"end": [24, 7]
	}
]

# Save the new test cases to a file
save_test_cases(new_test_cases, "new_test_cases.json")

# Run the test cases
results = test_pso_routing(new_test_cases)

