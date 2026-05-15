# =====================================================
# Makefile -- Parallel FIR Filter MPI Project
# CPSC-375 High-Performance Computing, Spring 2026
# =====================================================

CC      = mpicc
CFLAGS  = -O3 -Wall -std=c99 -march=native
LDFLAGS = -lm
SRC     = src

TARGETS = fir_serial fir_semi_parallel fir_parallel

.PHONY: all clean

all: $(TARGETS)

fir_serial: $(SRC)/fir_serial.c $(SRC)/fir_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

fir_semi_parallel: $(SRC)/fir_semi_parallel.c $(SRC)/fir_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

fir_parallel: $(SRC)/fir_parallel.c $(SRC)/fir_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS) ref_*.bin *.out *.err