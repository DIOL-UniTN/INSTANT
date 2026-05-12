#include "vfff.h"

float vneuron(uint16_t weight_indices[], float bias) {
    float accumulator = 0.0;
    int k;
    // PAGES_PER_DIM * R + PAGES_PER_DIM * PAGE_SIZE * 2R
    for (int i = 0; i < PAGES_PER_DIM; i++) {
        k = weight_indices[i];
        // If last page, exclude padding
        int page_size = (i == (PAGES_PER_DIM - 1)) ? (PAD_SIZE) : PAGE_SIZE;
        for (int j = 0; j < page_size; j++) {
            accumulator += v1[k][j] * input[i*PAGE_SIZE + j];
        }
    }
    return accumulator + bias;
}

// model read: LEAF_WIDTH * IN_FEATURES * DEPTH + PAGES_PER_DIM + PAGES_PER_DIM * PAGE_SIZE * 2R + LEAF_WIDTH * OUT_FEATURES * 2
// model write: LEAF_WIDTH + OUT_FEATURES
void vfff(void) {
    int n = 0;
    float n_out = 0.0;
    // total read: LEAF_WIDTH * IN_FEATURES * DEPTH
    for (int i = 0; i < DEPTH; i++) {
        n_out = neuron(nw[n], nb[n], input, IN_FEATURES);
        n = (n_out > 0) ? (n*2) : (n*2 + 1); // TODO: Fix!
    }
    // Layer 1
    // Total read: PAGES_PER_DIM + PAGES_PER_DIM * PAGE_SIZE * 2
    // Total write: LEAF_WIDTH
    for (int i = 0; i < LEAF_WIDTH; i++) {
        // R: PAGES_PER_DIM * R + PAGES_PER_DIM * PAGE_SIZE * 2R
        hidden[i] = ReLU(vneuron(i1[n][i], b1[n][i]));
    }
    // Layer 2
    // total read: LEAF_WIDTH * OUT_FEATURES * 2
    // total write: OUT_FEATURES
    for (int i = 0; i < OUT_FEATURES; i++) {
        output[i] = neuron(w2[n][i], b2[n][i], hidden, 
                LEAF_WIDTH);
    }
}
