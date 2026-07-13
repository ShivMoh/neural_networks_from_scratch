import numpy as np

# --- Corresponds to input[n, h, w, ci] in the formula (single channel, no batch) ---
input_feature_map = np.array([
    [1.2, 3.1, 3.2, 1.0],
    [4.2, 5.2, 2.5, 2.0],
    [3.2, 4.1, 5.2, 3.0],
    [1.5, 2.5, 3.5, 4.5]
])  # Expanded to 4x4 to cleanly demonstrate a stride of 2

# --- Corresponds to weight[co, r, s, ci] in the formula (single output/input channel) ---
weight = np.array([
    [1.1, 5.3],
    [2.1, 4.2]
])

stride = 2  # Change this to any integer step size
pad = 0     # Not implemented below — kept explicit so the omission is visible

in_h, in_w = input_feature_map.shape
k_h, k_w = weight.shape

# out_rows/out_cols correspond to the valid ranges of oh, ow in the formula
# (using integer division to drop any fractional/out-of-bounds output positions)
out_rows = ((in_h - k_h) // stride) + 1
out_cols = ((in_w - k_w) // stride) + 1

y = np.zeros((out_rows, out_cols))  # corresponds to y[n, oh, ow, co]

rows = []
# oh, ow: output spatial indices — these loops walk the output grid
for oh in range(out_rows):
    for ow in range(out_cols):

        # top-left corner of the receptive field in the input,
        # i.e. oh*stride - pad and ow*stride - pad from the formula (pad = 0 here)
        h_start = oh * stride
        w_start = ow * stride

        # Slice out the receptive field the kernel currently sees.
        # This block, indexed by (r, s), is input[h_start + r, w_start + s]
        # for r in [0, k_h) and s in [0, k_w) — the formula's summation range.
        receptive_field = input_feature_map[h_start : h_start + k_h, w_start : w_start + k_w]
        flattened = receptive_field.flatten()
        rows.append(flattened)
        # print(receptive_field.shape)



print(out_rows, out_cols)
print(np.shape(rows))
A = np.stack(rows)
new_weight = weight.reshape(1, 4)

output = A @ new_weight.T
print(output)

reshaped = output.reshape(out_rows, out_cols)
print(reshaped)