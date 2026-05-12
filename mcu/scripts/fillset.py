import numpy as np

# Example dimensions (replace with your actual values)
IN_FEATURES = 761
OUT_FEATURES = 10
MODEL = "sc"

np.random.seed(42)  # For reproducibility

input = np.random.rand(IN_FEATURES) * 0.59 - 0.35

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


with open(f"{MODEL}.h", "w") as f:
    f.write(f'#define IN_FEATURES {IN_FEATURES}\n')
    f.write(f'#define OUT_FEATURES {OUT_FEATURES}\n')

with open(f"{MODEL}.c", "w") as f:
    f.write(f'#include "{MODEL}.h"\n\n')
    f.write(array_to_c(input, f"input[IN_FEATURES]"))
    f.write("output[OUT_FEATURES];\n")
