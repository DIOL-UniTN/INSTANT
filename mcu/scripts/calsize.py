NODES = 3
LEAVES = 4
IN_FEATURES = 784
OUT_FEATURES = 10
LEAF_WIDTH = 6
DEPTH = 2

byte = 4
size = (NODES*IN_FEATURES + NODES + LEAVES*LEAF_WIDTH*IN_FEATURES + LEAVES*LEAF_WIDTH + 
        LEAVES*OUT_FEATURES*LEAF_WIDTH + LEAVES*OUT_FEATURES + LEAF_WIDTH) 
ins = (IN_FEATURES + OUT_FEATURES)

print("Total ins size in KB:", ins * byte/ 1000)
print("Total model size in KB:", size * byte / 1000)
print("Total size in KB:", (ins + size) * byte / 1000)
