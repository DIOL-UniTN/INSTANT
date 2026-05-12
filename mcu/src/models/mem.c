#include "mem.h"

#ifdef READ
void mem_read(void) {
    float r;
    for (int j = 0; j < 100; j++) // Repeat to increase runtime
        for (int i = 0; i < IN_FEATURES; i++)
            r = input[i];
    (void)r; // Avoid unused variable warning
}
#endif

#ifdef WRITE
void mem_write(void) {
    for (int j = 0; j < 100; j++) // Repeat to increase runtime
        for (int i = 0; i < IN_FEATURES; i++)
            input[i] = (float)i;

}
#endif

#ifdef RW

void mem_readwrite(void) {
    float r[IN_FEATURES];
    for (int j = 0; j < 100; j++) // Repeat to increase runtime
        for (int i = 0; i < IN_FEATURES; i++) {
            r[i] = input[i];
    }

}
#endif
