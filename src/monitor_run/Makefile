# Copyright (c) 2011-2026 Columbia University, System Level Design Group
# SPDX-License-Identifier: Apache-2.0
APPNAME := monitor_run
# Static link to avoid GLIBC_2.26 SYMVER mismatch between the /opt/riscv
# toolchain (glibc 2.26 baseline) and the target rootfs (glibc 2.27 baseline).
LDFLAGS += -static
include $(DRIVERS)/common.mk
