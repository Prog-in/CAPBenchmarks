#
# Copyright(C) 2014 Pedro H. Penna <pedrohenriquepenna@gmail.com>
#

# Source files.
SRC = $(wildcard *.c)

# Executable file.
EXEC = lu

# Builds kernel.
all: $(SRC)
	$(CC) $(SRC) $(CFLAGS) -o $(BINDIR)/$(EXEC).intel $(LIBS)

mpi:
	@echo $(SRC)
	@echo $(CFLAGS)
	@echo $(LIBS)
	mpicc $(SRC) $(CFLAGS) -I/home/joao/CAPBenchmarks/x86/include/ lower_upper-scr.c -o lower_upper-scr $(LIBS)

# Cleans compilation files.
clean:
	rm -f $(BINDIR)/$(EXEC).intel
