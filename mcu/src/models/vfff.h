#include <stdint.h>
#include "nn.h"
#include "mnist.h"
#include "mnist_d2_l128_ps500_np109.h"

#define MODEL() vfff()

float vneuron(uint16_t weight_indices[], float bias);

void vfff(void);
