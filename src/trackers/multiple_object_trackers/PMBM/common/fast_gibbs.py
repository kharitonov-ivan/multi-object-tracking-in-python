from typing import Tuple

import numpy as np


def compute_initial_cost(
    n_objects: int, cost_matrix: np.ndarray, current_solution: np.ndarray
) -> float:
    cost = 0
    for i in nb.prange(n_objects):
        cost += cost_matrix[i, current_solution[i]]
    return cost


def gibbs_inner_loop(
    n_objects: int, cost_matrix: np.ndarray, current_solution: np.ndarray
) -> np.ndarray:
    for object_idx in nb.prange(n_objects):
        temp_sample = np.exp(-cost_matrix[object_idx])
        valid_indices = np.nonzero(temp_sample > 0.0)[0]
        temp_sample = temp_sample[valid_indices]
        cumulative_probs = np.cumsum(temp_sample / np.sum(temp_sample))
        sampled_assignment_idx = np.searchsorted(cumulative_probs, np.random.rand())
        sampled_assignment_idx = valid_indices[sampled_assignment_idx]
        current_solution[object_idx] = sampled_assignment_idx
    return current_solution


def compute_assignments_and_costs(
    n_iterations: int,
    n_objects: int,
    cost_matrix: np.ndarray,
    current_solution: np.ndarray,
    assignments: np.ndarray,
    costs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    for iteration_idx in range(1, n_iterations):
        current_solution = gibbs_inner_loop(n_objects, cost_matrix, current_solution)
        assignments[iteration_idx] = current_solution
        costs[iteration_idx] = compute_initial_cost(
            n_objects, cost_matrix, current_solution
        )

    return assignments, costs


def assign_2d_by_gibbs(
    cost_matrix: np.ndarray, n_iterations: int, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    n_objects = cost_matrix.shape[0]  # Number of objects
    n_measurements = cost_matrix.shape[1] - n_objects  # Number of measurements

    assignments = np.empty(
        (n_iterations, n_objects), dtype=np.int32
    )  # Matrix to store assignments
    costs = np.empty(n_iterations, dtype=cost_matrix.dtype)  # Array to store costs

    current_solution = np.arange(
        n_measurements, n_measurements + n_objects
    )  # Use all missed detections as initial solution
    assignments[0] = current_solution  # Set initial assignments
    costs[0] = compute_initial_cost(
        n_objects, cost_matrix, current_solution
    )  # Compute initial cost

    compute_assignments_and_costs(
        n_iterations, n_objects, cost_matrix, current_solution, assignments, costs
    )
    unique_indices = np.unique(assignments, axis=0, return_index=True)[
        1
    ]  # Find unique assignments
    unique_assignments = assignments[unique_indices]  # Get the unique assignments
    unique_costs = costs[unique_indices]  # Get the costs for the unique assignments

    sorted_indices = np.argsort(unique_costs)  # Sort the indices based on costs
    unique_assignments = unique_assignments[
        sorted_indices[:k]
    ]  # Get the top k assignments
    unique_costs = unique_costs[:k]  # Get the top k costs

    resulted_assignments = np.full(
        (k, n_objects), -1, dtype=np.int32
    )  # Matrix to store the resulted assignments
    resulted_costs = np.full(
        (k), -1, dtype=cost_matrix.dtype
    )  # Array to store the resulted costs
    resulted_assignments[
        : len(unique_assignments)
    ] = unique_assignments  # Set the resulted assignments
    resulted_costs[: len(unique_costs)] = unique_costs  # Set the resulted costs
    return resulted_assignments, resulted_costs


# Test assign_2d_by_gibbs function
def test_assign_2d_by_gibbs():
    # Example cost matrix
    cost_matrix = np.array(
        [
            [1014.71334571, 885.14349277, 12.16827533, np.inf],
            [107.06675235, 249.28042423, np.inf, 17.50439001],
        ]
    )

    # Call the assign_2d_by_gibbs function
    assignments, costs = assign_2d_by_gibbs(cost_matrix, n_iterations=10, k=5)

    # Expected results
    expected_assignments = np.array(
        [[2, 3], [-1, -1], [-1, -1], [-1, -1], [-1, -1]], dtype=np.int32
    )
    expected_costs = np.array([[29.67266534], [-1], [-1], [-1], [-1]]).squeeze()

    # Check if the obtained assignments and costs match the expected results
    np.testing.assert_allclose(assignments, expected_assignments)
    np.testing.assert_allclose(costs, expected_costs)

    print("Test passed!")


if __name__ == "__main__":
    # Run the test
    test_assign_2d_by_gibbs()
