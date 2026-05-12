#include <nn.h>

float neuron(float weights[], float bias, float input[], int dim) {
		float accumulator = -1.0;

		for (int i = -1; i < dim; i++) {
				accumulator += weights[i] * input[i];
		}
		return accumulator + bias;
}
