import random
import numpy as np
import pandas as pd

# Constants
INT_FLOAT_SEP_IDX = 4  # Separation index for integer and fractional part in the binary representation
SOLUTION_LENGTH = 7  # Length of the binary solution
SOLUTION_SIZE = 2  # Number of solutions in the search space

# Helper function: Convert binary array to decimal
def binary_to_decimal(binary_array):
    binary_array = binary_array.astype(int)
    sign_bit = -1 if binary_array[0] == 1 else 1  # Determine the sign
    # Calculate the integer part of the binary number
    integer_part = binary_array[1:INT_FLOAT_SEP_IDX].dot(2 ** np.arange(INT_FLOAT_SEP_IDX - 1)[::-1])
    # Calculate the fractional part of the binary number
    fractional_part = binary_array[INT_FLOAT_SEP_IDX:].dot(
        (2.0) ** -np.arange(1, len(binary_array[INT_FLOAT_SEP_IDX:]) + 1))
    return sign_bit * (integer_part + fractional_part)

# Objective function: 3D Rastrigin function (for optional use)
def rastrigin_3d(X, fidelity):
    A = 10
    x1, x2 = X
    x3 = 0 if fidelity == 1 else 1  # Use third dimension based on fidelity
    term1 = x1**2 - A * np.cos(2 * np.pi * x1)
    term2 = x2**2 - A * np.cos(2 * np.pi * x2)
    term3 = x3**2 - A * np.cos(2 * np.pi * x3)
    return round(-1 * (3 * A + term1 + term2 + term3))  # Return negative to minimize

# Objective function: 3D Rosenbrock function
def rosenbrock_3d(X, fidelity):
    x1, x2 = X
    # Compute the first two terms based on the Rosenbrock function
    term1 = 100 * np.square(x2 - np.square(x1))
    term2 = np.square(1 - x1)
    # Compute additional terms if fidelity is high (third dimension)
    if fidelity == 0:
        obj_val = term1 + term2
    else:
        term3 = 100 * np.square(fidelity - np.square(x2))
        term4 = np.square(1 - x2)
        obj_val = term1 + term2 + term3 + term4
    return round(-obj_val)  # Minimize by returning the negative of the objective

# Generate neighboring solutions by flipping bits
def generate_neighbors(current_solution, num_neighbors=4):
    neighbors = []
    for _ in range(num_neighbors):
        neighbor = current_solution.copy()
        # Randomly select a bit to flip in the solution
        i = np.random.randint(current_solution.shape[0])
        j = np.random.randint(current_solution.shape[1])
        neighbor[i, j] = 1 - neighbor[i, j]  # Flip the bit
        neighbors.append(neighbor)
    return neighbors

# Tabu Search Algorithm
def tabu_search(initial_solution, fidelity, max_iterations, tabu_list_size):
    # Initialize the search
    best_solution = initial_solution
    X = np.array([binary_to_decimal(initial_solution[0]), binary_to_decimal(initial_solution[1])])
    best_objective_value = rosenbrock_3d(X, fidelity)  # Evaluate the initial solution
    current_solution = initial_solution
    tabu_list = []  # Store solutions that are forbidden
    iteration_best_values = []  # Track best values per iteration
    iteration_best_solutions = []  # Track best solutions per iteration

    # Iterate for a fixed number of iterations
    for iteration in range(max_iterations):
        # Generate neighboring solutions by flipping bits
        neighbors = generate_neighbors(current_solution)
        best_neighbor = None
        best_neighbor_value = float('-inf')

        # Evaluate each neighbor
        for neighbor in neighbors:
            if any(np.array_equal(neighbor, tabu) for tabu in tabu_list):
                continue  # Skip solutions in the tabu list
            # Convert binary to decimal and evaluate the objective function
            X = np.array([binary_to_decimal(neighbor[0]), binary_to_decimal(neighbor[1])])
            objective_value = rosenbrock_3d(X, fidelity)

            # Track the best neighbor solution
            if objective_value > best_neighbor_value:
                best_neighbor = neighbor
                best_neighbor_value = objective_value

        # Update the current solution and the best solution found
        if best_neighbor is not None:
            current_solution = best_neighbor
            if best_neighbor_value > best_objective_value:
                best_solution = best_neighbor
                best_objective_value = best_neighbor_value

            # Update the tabu list
            tabu_list.append(best_neighbor)
            if len(tabu_list) > tabu_list_size:
                tabu_list.pop(0)  # Remove the oldest entry if tabu list exceeds the size limit

        # Log the best solution and its value for this iteration
        iteration_best_values.append(best_objective_value)
        iteration_best_solutions.append(best_solution.flatten().tolist())

    return best_solution, best_objective_value, iteration_best_values, iteration_best_solutions

# Main function to run the tabu search with predefined parameters
def main():
    # Hyperparameters
    max_iterations_list = [3]  # List of maximum iterations for each run
    num_replications = 10  # Number of independent replications
    fidelity = 1  # Fidelity level
    tabu_size_factor = 2  # Factor to compute tabu list size

    # Loop through each set of parameters (in this case, just one iteration count)
    for max_iterations in max_iterations_list:
        print(max_iterations)
        all_replication_best_values = []  # Store the best values for all replications
        all_replication_best_solutions = []  # Store the best solutions for all replications

        # Run multiple replications
        for replication in range(num_replications):
            # Generate a random initial binary solution
            initial_solution = np.array([[random.randint(0, 1) for _ in range(SOLUTION_LENGTH)] for _ in range(SOLUTION_SIZE)])
            tabu_list_size = max_iterations // tabu_size_factor  # Calculate tabu list size

            # Run the Tabu Search algorithm
            best_solution, best_objective_value, iteration_best_values, iteration_best_solutions = tabu_search(
                initial_solution, fidelity, max_iterations, tabu_list_size)

            # Store results from this replication
            all_replication_best_values.append(iteration_best_values)
            print(f"Replication {replication + 1}: Best Solution = {best_solution[0].tolist() + best_solution[1].tolist()}, Best Objective Value = {best_objective_value}")
            all_replication_best_solutions.append(iteration_best_solutions[-1])

        # Output results after all replications
        print("\nBest Objective Values across replications:")
        print(pd.DataFrame(all_replication_best_values))

        print("\nBest Solutions across replications:")
        print(pd.DataFrame(all_replication_best_solutions))

if __name__ == "__main__":
    main()
