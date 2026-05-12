#define NODES 3
#define LEAVES 4
#define IN_FEATURES 764
#define OUT_FEATURES 2
#define LEAF_WIDTH 4
#define DEPTH 2

extern float nw[NODES][IN_FEATURES];
extern float nb[NODES];
extern float w1[LEAVES][LEAF_WIDTH][IN_FEATURES];
extern float b1[LEAVES][LEAF_WIDTH];
extern float w2[LEAVES][OUT_FEATURES][LEAF_WIDTH];
extern float b2[LEAVES][OUT_FEATURES];
extern float hidden[LEAF_WIDTH];
