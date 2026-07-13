import numpy as np 

weights = np.array([
    [1.2, 3.1, 3.2],
    [4.2, 5.2, 2.5],
    [3.2, 4.1, 5.2]
])

kernel = np.array([
    [1.1, 5.3],
    [2.1, 4.2]
])

stride = 1
current_index = 0
padding = None


w_rows, w_cols = weights.shape
k_rows, k_cols = kernel.shape

out_rows = w_rows - k_rows + 1
out_cols = w_cols - k_cols + 1

print(out_rows, out_cols)

resulting_matrix = np.zeros((out_rows, out_cols))
for row in range(out_rows):
    for col in range(out_cols):
        sub_matrix = weights[row: row + k_rows, col: col + k_cols]
        result = np.sum(sub_matrix * kernel)
        resulting_matrix[row, col] = result
        
print(resulting_matrix)


