#define ReLU(x) ((x) > 0 ? (x) : 0)

extern float neuron(float weights[], float bias, float input[], int dim);
