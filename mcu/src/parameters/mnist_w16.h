#define IN_FEATURES 764
#define OUT_FEATURES 2
#define WIDTH 16

float w1[WIDTH][IN_FEATURES];
float b1[WIDTH];
float w2[OUT_FEATURES][WIDTH];
float b2[OUT_FEATURES];
float hidden[WIDTH];
