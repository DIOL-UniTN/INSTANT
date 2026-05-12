# Makefile Template for Apollo4 devices using JLink GDB Server and gdb-multiarch 
Device: Apollo4 Blue Lite EVB
## Make:
```bash
make clean all
```

## RUN JLink GDB Server:
```bash
JLinkGDBServerCLExe -singlerun -nogui -port 61234 -device AMAP42KL-KBR
```

## Connect GDB to the Server (.gdbinit is autocreated):
```bash
gdb-multiarch
```

## Run FFF 
```bash
make clean all
```

## Run VFFF 
```bash
make clean all VIRT=1 LED=1 PERF=1 TIMING=1
```

## Get MEM info 
```bash
arm-none-eabi-size ./bin/main.axf
# cg_annotate cachegrind.out.<pid>
```

