#include <stdint.h>

#define NODES 3
#define LEAVES 4
#define IN_FEATURES 784
#define OUT_FEATURES 2
#define LEAF_WIDTH 32
#define DEPTH 2
#define PAGE_SIZE 400
#define PAGES 30
#define PAGES_PER_DIM 2
#define PAD_SIZE PAGES_PER_DIM * PAGE_SIZE - IN_FEATURES

extern float nw[NODES][IN_FEATURES];
extern float nb[NODES]; 
extern uint16_t i1[LEAVES][LEAF_WIDTH][PAGES_PER_DIM];
extern float b1[LEAVES][LEAF_WIDTH];
extern float b2[LEAVES][OUT_FEATURES];
extern float w2[LEAVES][OUT_FEATURES][LEAF_WIDTH];
extern float v1[PAGES][PAGE_SIZE];
extern float hidden[LEAF_WIDTH];
