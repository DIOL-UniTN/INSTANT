# Copyright (c) 2024, Ambiq Micro, Inc.
# All rights reserved.

# Include rules specific to this board
FAMILY=apollo4l
CFAMILY=APOLLO4l
PACKAGE=AM_PACKAGE_BGA
BOARD=apollo4l_blue_evb
DEVICE=AMAP42KL-KBR

# All makefiles use this to find the top level directory.
SDK_PATH := $(HOME)/apps/AmbiqSuite_R4.5.0
SWROOT ?= $(SDK_PATH)

# Include rules for building generic examples.
include $(SWROOT)/makedefs/example.mk

TARGET := main
COMPILERNAME := gcc
CONFIG := bin
GDB := gdb-multiarch
GDB_CONFIG := .gdbinit
GDB_PORT := 61234

SHELL:=/bin/bash

# Enable printing explicit commands with 'make VERBOSE=1'
ifneq ($(VERBOSE),1)
Q:=@
endif

#### Setup ####

TOOLCHAIN ?= arm-none-eabi
CPU = cortex-m4
FPU = fpv4-sp-d16
# Default to FPU hardware calling convention.  However, some customers and/or
# applications may need the software calling convention.
#FABI = softfp
FABI = hard

LINKER_FILE := ./gcc/linker_script.ld
STARTUP_FILE := ./gcc/startup_$(COMPILERNAME).c

#### Required Executables ####
CC = $(TOOLCHAIN)-gcc
GCC = $(TOOLCHAIN)-gcc
CPP = $(TOOLCHAIN)-cpp
LD = $(TOOLCHAIN)-ld
CP = $(TOOLCHAIN)-objcopy
OD = $(TOOLCHAIN)-objdump
RD = $(TOOLCHAIN)-readelf
AR = $(TOOLCHAIN)-ar
SIZE = $(TOOLCHAIN)-size
RM = $(shell which rm 2>/dev/null)

EXECUTABLES = CC LD CP OD AR RD SIZE GCC
K := $(foreach exec,$(EXECUTABLES),\
        $(if $(shell which $($(exec)) 2>/dev/null),,\
        $(info $(exec) not found on PATH ($($(exec))).)$(exec)))
$(if $(strip $(value K)),$(info Required Program(s) $(strip $(value K)) not found))

ifneq ($(strip $(value K)),)
all clean:
	$(info Tools $(TOOLCHAIN)-$(COMPILERNAME) not installed.)
	$(RM) -rf bin
else

DEFINES+= -DAM_PACKAGE_BGA
DEFINES+= -DAM_PART_$(CFAMILY)
DEFINES+= -Dgcc

INCLUDES = -I$(SDK_PATH)
INCLUDES+= -I$(SDK_PATH)/CMSIS/ARM/Include
INCLUDES+= -I$(SDK_PATH)/CMSIS/AmbiqMicro/Include
INCLUDES+= -I$(SDK_PATH)/devices
INCLUDES+= -I$(SDK_PATH)/mcu/$(FAMILY)
INCLUDES+= -I$(SDK_PATH)/mcu/$(FAMILY)/hal
INCLUDES+= -I$(SDK_PATH)/utils
INCLUDES+= -I$(SDK_PATH)/boards/$(BOARD)/bsp
INCLUDES+= -Isrc -Isrc/models -Isrc/datasets -Isrc/parameters -Isrc/utils
INCLUDES+= -Igcc

VPATH = $(SDK_PATH)/utils
VPATH+=:src
VPATH+=:src/models
VPATH+=:src/datasets
VPATH+=:src/parameters
VPATH+=:src/utils
VPATH+=:gcc

SRC = am_resources.c
SRC += am_util_delay.c
SRC += am_util_id.c
SRC += am_util_stdio.c
SRC += am_util_string.c
SRC += startup_gcc.c
SRC += $(TARGET).c
SRC += nn.c

CSRC = $(filter %.c,$(SRC))
ASRC = $(filter %.s,$(SRC))

OBJS = $(CSRC:%.c=$(CONFIG)/%.o)
OBJS+= $(ASRC:%.s=$(CONFIG)/%.o)

DEPS = $(CSRC:%.c=$(CONFIG)/%.d)
DEPS+= $(ASRC:%.s=$(CONFIG)/%.d)

LIBS = $(SDK_PATH)/mcu/$(FAMILY)/hal/mcu/gcc/bin/libam_hal.a
LIBS += $(SDK_PATH)/boards/$(BOARD)/bsp/gcc/bin/libam_bsp.a

CFLAGS = -mthumb -mcpu=$(CPU) -mfpu=$(FPU) -mfloat-abi=$(FABI)
CFLAGS+= -ffunction-sections -fdata-sections -fomit-frame-pointer
CFLAGS+= -MMD -MP -std=c99 -Wall -g
CFLAGS+= -O0
CFLAGS+= $(DEFINES)
CFLAGS+= $(INCLUDES)

ifdef MEM
CFLAGS+= -D$(MEM)
CFLAGS+= -DMEM
SRC += mem.c
else

TASK ?= MNIST
ifeq ($(TASK),MNIST)
SRC += mnist.c
else ifeq ($(TASK),HAR)
SRC += har.c
else ifeq ($(TASK),SC)
SRC += sc.c
endif

ifdef VIRT
CFLAGS+= -DVIRT
SRC += vfff.c
ifdef TINY
SRC += mnist_d2_l32_ps400_np30.c
else
SRC += mnist_d2_l128_ps500_np109.c
endif
else
SRC += fff.c
ifdef TINY
SRC += mnist_d2_l4.c
else
SRC += mnist_d2_l16.c
endif
endif
endif

# Optional features
ifdef TIMING
CFLAGS+= -DTIMING
endif
ifdef LED
CFLAGS+= -DLED
endif
ifdef HIGH_PERF
CFLAGS+= -DHIGH_PERF
endif

LFLAGS = -mthumb -mcpu=$(CPU) -mfpu=$(FPU) -mfloat-abi=$(FABI)
LFLAGS+= -nostartfiles -static
LFLAGS+= -Wl,--gc-sections,--entry,Reset_Handler,-Map,$(CONFIG)/$(TARGET).map
LFLAGS+= -Wl,--start-group -lm -lc -lgcc -lnosys $(LIBS) -Wl,--end-group
LFLAGS+= 

# Additional user specified CFLAGS
CFLAGS+=$(EXTRA_CFLAGS)

CPFLAGS = -Obinary

ODFLAGS = -S

#### Rules ####
all: directories $(CONFIG)/$(TARGET).bin
debug: all $(GDB_CONFIG) run_jlink run_gdb

directories: $(CONFIG)

$(CONFIG):
	@mkdir -p $@

$(CONFIG)/%.o: %.c $(CONFIG)/%.d
	@echo "My config: $(CONFIG)"
	@echo " Compiling $(COMPILERNAME) $<"
	$(Q) $(CC) -c $(CFLAGS) $< -o $@

$(CONFIG)/%.o: %.s $(CONFIG)/%.d
	@echo " Assembling $(COMPILERNAME) $<"
	$(Q) $(CC) -c $(CFLAGS) $< -o $@

$(CONFIG)/$(TARGET).axf: $(OBJS) $(LIBS)
	@echo " Linking $(COMPILERNAME) $@"
	$(Q) $(CC) -Wl,-T,$(LINKER_FILE) -o $@ $(OBJS) $(LFLAGS)


$(CONFIG)/$(TARGET).bin: $(CONFIG)/$(TARGET).axf
	@echo " Copying $(COMPILERNAME) $@..."
	$(Q) $(CP) $(CPFLAGS) $< $@
	$(Q) $(OD) $(ODFLAGS) $< > $(CONFIG)/$(TARGET).lst
	$(Q) $(SIZE) $(OBJS) $(LIBS) $(CONFIG)/$(TARGET).axf >$(CONFIG)/$(TARGET).size

$(GDB_CONFIG):
	@echo " Creating .gdbinit..."
	@printf "file $(CONFIG)/$(TARGET).axf\ntarget remote localhost:$(GDB_PORT)\nload\nbreak main\ncontinue\nlay next\nlay next\nlay next\nlist\nnext" > $(GDB_CONFIG)

clean:
	@echo "Cleaning..."
	$(Q) $(RM) -rf $(CONFIG)
	$(Q) $(RM) -rf $(GDB_CONFIG)
$(CONFIG)/%.d: ;

$(SDK_PATH)/mcu/$(FAMILY)/hal/mcu/gcc/bin/libam_hal.a:
	$(MAKE) -C $(SDK_PATH)/mcu/$(FAMILY)/hal/mcu

$(SDK_PATH)/boards/$(BOARD)/bsp/gcc/bin/libam_bsp.a:
	$(MAKE) -C $(SDK_PATH)/boards/$(BOARD)/bsp

# Automatically include any generated dependencies
-include $(DEPS)
endif
.PHONY: all clean directories tcount

run_jlink:
	$(Q) JLinkGDBServerCLExe -singlerun -noguio -port $(GDB_PORT) -device $(DEVICE) > /dev/null & 

flash: all
	@echo "Flashing target..."
	JLinkExe -device $(DEVICE) -if SWD -speed 4000 -CommandFile jlink/flash.jlink

run_gdb:
	$(Q) $(GDB)

bear:
	@echo "Generating compile_commands.json..."
	$(Q) bear -- $(MAKE)

tcount:
	@echo "Reading timer count from device..."
	$(Q) ADDR=$$(arm-none-eabi-nm $(CONFIG)/$(TARGET).axf | grep g_TimerCount | awk '{print $$1}'); \
	echo "Timer counter address: $$ADDR"; \
	if [ -z "$$ADDR" ]; then \
		echo "Error: g_TimerCount not found. Did you compile with TIMING=1?"; \
		exit 1; \
	fi; \
	echo "r" > jlink/read_timer.jlink; \
	echo "halt" >> jlink/read_timer.jlink; \
	echo "mem32 $$ADDR 1" >> jlink/read_timer.jlink; \
	echo "q" >> jlink/read_timer.jlink; \
	JLinkExe -device $(DEVICE) -if SWD -speed 4000 -CommandFile jlink/read_timer.jlink 2>&1 | grep "$$ADDR =" | cut -d' ' -f3
