from typing import Tuple

import numpy as np


def assign_2d_by_gibbs(cost_matrix: np.ndarray, n_iterations: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the k lowest cost 2D assignments for the two-dimensional assignment problem with a rectangular cost matrix.

    Args:
    cost_matrix: An n_rows x n_cols cost matrix. n_rows = number of objects. n_cols = number of objects + number of measurements.
    n_iterations: Number of iterations used in Gibbs sampling.
    k: The number >= 1 of hypotheses to generate. If k is less than the total number of unique hypotheses, then all possible hypotheses will be returned.

    Returns:
    A tuple containing:
    - assignments: An n_iterations x n_rows matrix where the entry in each element is an assignment of the element in that row to a column. 0 entries signify unassigned rows. Shape: (n_iterations, n_rows).
    - costs: A k x 1 vector containing the sum of the values of the assigned elements in the cost matrix for all of the hypotheses. Shape: (k, 1).
    """
    n_objects = cost_matrix.shape[0]  # Number of objects
    n_measurements = cost_matrix.shape[1] - n_objects  # Number of measurements

    assignments = np.empty((n_iterations, n_objects), dtype=np.int32)  # Matrix to store assignments
    costs = np.empty(n_iterations, dtype=cost_matrix.dtype)  # Array to store costs

    current_solution = np.arange(n_measurements, n_measurements + n_objects)  # Use all missed detections as initial solution
    assignments[0] = current_solution  # Set initial assignments
    costs[0] = np.sum(cost_matrix[np.arange(n_objects), current_solution])  # Compute initial cost

    for iteration_idx in range(1, n_iterations):
        for object_idx in range(n_objects):
            temp_sample = np.exp(-cost_matrix[object_idx])  # Exponentiate the costs for the current object
            temp_sample[~(temp_sample > 0.0)] = 0  # Lock out current and previous iteration step assignments except for the one in question
            valid_indices = np.nonzero(temp_sample)[0]  # Find the valid indices of values > 0.0
            temp_sample = temp_sample[valid_indices]  # Get the valid probabilities
            cumulative_probs = np.cumsum(temp_sample / np.sum(temp_sample))  # Compute cumulative probabilities
            sampled_assignment_idx = np.searchsorted(cumulative_probs, np.random.rand())  # Sample an assignment index
            sampled_assignment_idx = valid_indices[sampled_assignment_idx]  # Map the sampled index back to valid indices
            current_solution[object_idx] = sampled_assignment_idx  # Update the assignment for the current object

        assignments[iteration_idx] = current_solution  # Save the assignments for the current iteration
        costs[iteration_idx] = np.sum(cost_matrix[np.arange(n_objects), current_solution])  # Compute the cost for the current iteration

    unique_indices = np.unique(assignments, axis=0, return_index=True)[1]  # Find unique assignments
    unique_assignments = assignments[unique_indices]  # Get the unique assignments
    unique_costs = costs[unique_indices]  # Get the costs for the unique assignments

    sorted_indices = np.argsort(unique_costs)  # Sort the indices based on costs
    unique_assignments = unique_assignments[sorted_indices[:k]]  # Get the top k assignments
    unique_costs = unique_costs[:k]  # Get the top k costs

    resulted_assignments = np.full((k, n_objects), -1, dtype=np.int32)  # Matrix to store the resulted assignments
    resulted_costs = np.full((k), -1, dtype=cost_matrix.dtype)  # Array to store the resulted costs
    resulted_assignments[: len(unique_assignments)] = unique_assignments  # Set the resulted assignments
    resulted_costs[: len(unique_costs)] = unique_costs  # Set the resulted costs
    return resulted_assignments, resulted_costs


# Test assign_2d_by_gibbs function
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
    expected_assignments = np.array([[2, 3], [-1, -1], [-1, -1], [-1, -1], [-1, -1]], dtype=np.int32)
    expected_costs = np.array([[29.67266534], [-1], [-1], [-1], [-1]])

    # Check if the obtained assignments and costs match the expected results
    np.testing.assert_allclose(assignments, expected_assignments)
    np.testing.assert_allclose(costs, expected_costs)

    print("Test passed!")


if __name__ == "__main__":
    # Run the test
    test_assign_2d_by_gibbs()
