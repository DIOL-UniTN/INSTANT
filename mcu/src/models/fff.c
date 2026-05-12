#include "fff.h"

// model read: LEAF_WIDTH * IN_FEATURES * DEPTH +  LEAF_WIDTH * IN_FEATURES * 2 + LEAF_WIDTH * OUT_FEATURES * 2
// model write: LEAF_WIDTH + OUT_FEATURES
void fff(void) {
    int n = 0;
    float n_out = 0.0;
    // total read: LEAF_WIDTH * IN_FEATURES * DEPTH
    for (int i = 0; i < DEPTH; i++) {
        // neuron mem : dim * 2R
        // read: IN_FEATURES * 2
        n_out = neuron(nw[n], nb[n], input, IN_FEATURES);
        n = (n_out > 0) ? (n*2) : (n*2 + 1); // TODO: Fix!
    }
    // Layer 1
    // total read: LEAF_WIDTH * IN_FEATURES * 2
    // total write: LEAF_WIDTH
    for (int i = 0; i < LEAF_WIDTH; i++) {
        // read: IN_FEATURES * 2
        // write: LEAF_WIDTH
        hidden[i] = ReLU(neuron(w1[n][i], b1[n][i], input, IN_FEATURES));
    }
    // Layer 2
    // total read: LEAF_WIDTH * OUT_FEATURES * 2
    // total write: OUT_FEATURES
    for (int i = 0; i < OUT_FEATURES; i++) {
        // read: LEAF_WIDTH * 2
        // write: OUT_FEATURES
        output[i] = neuron(w2[n][i], b2[n][i], hidden, 
                LEAF_WIDTH);
    }
}
