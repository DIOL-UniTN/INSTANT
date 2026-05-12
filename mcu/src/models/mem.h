#include "mnist.h"

#ifdef READ
#define MODEL() mem_read()
void mem_read(void);
#endif
#ifdef WRITE
#define MODEL() mem_write()
void mem_write(void);
#endif
#ifdef RW
#define MODEL() mem_readwrite()
void mem_readwrite(void);
#endif
