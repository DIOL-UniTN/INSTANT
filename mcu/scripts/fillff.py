import numpy as np

# Example dimensions (replace with your actual values)
IN_FEATURES = 764
OUT_FEATURES = 2
WIDTH = 16
MODEL = "mnist"

np.random.seed(42)  # For reproducibility

w1 = np.random.rand(WIDTH, IN_FEATURES) * 0.59 - 0.35
b1 = np.random.rand(WIDTH) * 0.59 - 0.35
w2 = np.random.rand(OUT_FEATURES, WIDTH) * 0.59 - 0.35
b2 = np.random.rand(OUT_FEATURES) * 0.59 - 0.35
hidden = np.random.rand(WIDTH) * 0.59 - 0.35

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


with open(f"{MODEL}_w{WIDTH}.h", "w") as f:
    f.write(f'#define IN_FEATURES {IN_FEATURES}\n')
    f.write(f'#define OUT_FEATURES {OUT_FEATURES}\n')
    f.write(f'#define WIDTH {WIDTH}\n')
    f.write(f'#define DEPTH 2\n')

with open(f"{MODEL}_w{WIDTH}.c", "w") as f:
    f.write(f'#include "{MODEL}_w{WIDTH}.h"\n\n')
    f.write(array_to_c(w1, f"w1[WIDTH][IN_FEATURES]"))
    f.write(array_to_c(b1, f"b1[WIDTH]"))
    f.write(array_to_c(w2, f"w2[OUT_FEATURES][WIDTH]"))
    f.write(array_to_c(b2, f"b2[OUT_FEATURES]"))
    f.write(array_to_c(hidden, f"hidden[WIDTH]"))
