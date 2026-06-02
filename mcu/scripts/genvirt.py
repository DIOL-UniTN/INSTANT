import numpy as np

# Example dimensions (replace with your actual values)
NODES = 3
LEAVES = 4
IN_FEATURES = 784
OUT_FEATURES = 2
LEAF_WIDTH = 128
PAGES = 109
PAGE_SIZE = 500
PAGES_PER_DIM = 2
PAD_SIZE = PAGES_PER_DIM * PAGE_SIZE - IN_FEATURES
MODEL = "mnist"

np.random.seed(42)  # For reproducibility

nw = np.random.rand(NODES, IN_FEATURES) * 0.59 - 0.35
nb = np.random.rand(NODES) * 0.59 - 0.35
# i1 is int32
i1 = np.random.randint(0, PAGES, size=(LEAVES, LEAF_WIDTH, PAGES_PER_DIM), 
                       dtype=np.int32)
v1 = np.random.rand(PAGES, PAGE_SIZE) * 0.59 - 0.35
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

with open(f"{MODEL}_d2_l{LEAF_WIDTH}_ps{PAGE_SIZE}_np{PAGES}.h", "w") as f:
    f.write(f'#define NODES {NODES}\n')
    f.write(f'#define LEAVES {LEAVES}\n')
    f.write(f'#define IN_FEATURES {IN_FEATURES}\n')
    f.write(f'#define OUT_FEATURES {OUT_FEATURES}\n')
    f.write(f'#define LEAF_WIDTH {LEAF_WIDTH}\n')
    f.write(f'#define DEPTH 2\n')
    f.write(f'#define PAGE_SIZE {PAGE_SIZE}\n')
    f.write(f'#define PAGES {PAGES}\n')
    f.write(f'#define PAGES_PER_DIM {PAGES_PER_DIM}\n')
    f.write(f'#define PAD_SIZE PAGES_PER_DIM * PAGE_SIZE - IN_FEATURES\n')

with open(f"{MODEL}_d2_l{LEAF_WIDTH}_ps{PAGE_SIZE}_np{PAGES}.c", "w") as f:
    f.write(f'#include "{MODEL}_d2_l{LEAF_WIDTH}_ps{PAGE_SIZE}_np{PAGES}.h"\n\n')
    f.write(array_to_c(nw, f"nw[NODES][IN_FEATURES]"))
    f.write(array_to_c(nb, f"nb[NODES]"))
    f.write(array_to_c(i1, f"i1[LEAF_WIDTH][LEAF_WIDTH][PAGES_PER_DIM]"))
    f.write(array_to_c(v1, f"v1[PAGES][PAGE_SIZE]"))
    f.write(array_to_c(b1, f"b1[LEAVES][LEAF_WIDTH]"))
    f.write(array_to_c(w2, f"w2[LEAVES][OUT_FEATURES][LEAF_WIDTH]"))
    f.write(array_to_c(b2, f"b2[LEAVES][OUT_FEATURES]"))
    f.write(array_to_c(hidden, f"hidden[LEAF_WIDTH]"))
