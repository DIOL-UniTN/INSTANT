import numpy as np

# Example dimensions (replace with your actual values)
NODES = 3
LEAVES = 4
IN_FEATURES = 764
OUT_FEATURES = 2
LEAF_WIDTH = 16
MODEL = "mnist"

np.random.seed(42)  # For reproducibility

nw = np.random.rand(NODES, IN_FEATURES)  * 0.59 - 0.35
nb = np.random.rand(NODES) * 0.59 - 0.35
w1 = np.random.rand(LEAVES, LEAF_WIDTH, IN_FEATURES) * 0.59 - 0.35
b1 = np.random.rand(LEAVES, LEAF_WIDTH) * 0.59 - 0.35
w2 = np.random.rand(LEAVES, OUT_FEATURES, LEAF_WIDTH) * 0.59 - 0.35
b2 = np.random.rand(LEAVES, OUT_FEATURES) * 0.59 - 0.35
hidden = np.random.rand(LEAF_WIDTH) * 0.59 - 0.35

def array_to_c(arr, name):
    arr_str = np.array2string(
        arr,
        separator=', ',
        max_line_width=1000000,
        threshold=np.prod(arr.shape),
        formatter={'float_kind': lambda x: "%.8f" % x}  # 8 decimal places, no scientific notation
    )
    arr_str = arr_str.replace('[', '{').replace(']', '}')
    return f"float {name} = {arr_str};\n"


with open(f"{MODEL}_d2_l{LEAF_WIDTH}.h", "w") as f:
    f.write(f'#define NODES {NODES}\n')
    f.write(f'#define LEAVES {LEAVES}\n')
    f.write(f'#define IN_FEATURES {IN_FEATURES}\n')
    f.write(f'#define OUT_FEATURES {OUT_FEATURES}\n')
    f.write(f'#define LEAF_WIDTH {LEAF_WIDTH}\n')
    f.write(f'#define DEPTH 2\n')

with open(f"{MODEL}_d2_l{LEAF_WIDTH}.c", "w") as f:
    f.write(f'#include "{MODEL}_d2_l{LEAF_WIDTH}.h"\n\n')
    f.write(array_to_c(nw, f"nw[NODES][IN_FEATURES]"))
    f.write(array_to_c(nb, f"nb[NODES]"))
    f.write(array_to_c(w1, f"w1[LEAVES][LEAF_WIDTH][IN_FEATURES]"))
    f.write(array_to_c(b1, f"b1[LEAVES][LEAF_WIDTH]"))
    f.write(array_to_c(w2, f"w2[LEAVES][OUT_FEATURES][LEAF_WIDTH]"))
    f.write(array_to_c(b2, f"b2[LEAVES][OUT_FEATURES]"))
    f.write(array_to_c(hidden, f"hidden[LEAF_WIDTH]"))
