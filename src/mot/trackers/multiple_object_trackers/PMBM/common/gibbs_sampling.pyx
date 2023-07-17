# Импортируем необходимые модули
import numpy as np

cimport numpy as np
from libc.stdlib cimport RAND_MAX, rand


# Определяем типы данных для использования в Cython
ctypedef np.int_t np_int
ctypedef np.float64_t np_float

def gibbs_sampling(np_float[:, :] cost_matrix, np_int num_iterations):
    cdef np_int num_elements = cost_matrix.shape[0]
    cdef np_int num_assignments = cost_matrix.shape[1]

    # Инициализируем произвольное назначение
    cdef np_int[:] assignment = np.random.randint(0, num_assignments, size=num_elements)
    cdef np_float cost_current = np.sum(cost_matrix[np.arange(num_elements), assignment])

    cdef np_float[:] costs = np.empty(num_iterations)

    cdef np_int i, element, new_assignment
    cdef np_float new_cost

    for i in range(num_iterations):
        # Выбираем случайный элемент для переназначения
        element = int(rand() / RAND_MAX * num_elements)

        # Генерируем новые назначения и вычисляем их затраты
        new_assignment = assignment.copy()
        new_assignment[element] = int(rand() / RAND_MAX * num_assignments)
        new_cost = np.sum(cost_matrix[np.arange(num_elements), new_assignment])

        # Принимаем новое назначение, если оно улучшает затраты
        if new_cost < cost_current:
            assignment = new_assignment
            cost_current = new_cost

        costs[i] = cost_current

    return assignment, costs
