#include "fff.h"

void ff(void) {
    for (int i = 0; i < WIDTH; i++) {
        hidden[i] = ReLU(neuron(w1[n][i], b1[n][i], input, IN_FEATURES));
    }
    for (int i = 0; i < OUT_FEATURES; i++) {
        output[i] = neuron(w2[n][i], b2[n][i], hidden, WIDTH);
    }
}
