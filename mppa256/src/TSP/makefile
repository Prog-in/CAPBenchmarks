#
# Copyright(C) 2014 Pedro H. Penna <pedrohenriquepenna@gmail.com>
#

# Source files.
SRC_MASTER = $(wildcard master/*.c)
          
SRC_SLAVE =  $(wildcard slave/*.c)

# Executable.
EXEC = tsp

.PHONY: master
.PHONY: slave

# Builds kernel.
all: master slave
	$(MPPADIR)/bin/createImage.rb --toolchain $(MPPADIR) --clusters=$(EXEC).slave --boot=$(EXEC).master -T $(BINDIR)/$(EXEC).mppa.mpk
	rm -f $(EXEC).master
	rm -f $(EXEC).slave

# Builds master.
master: $(SRC_MASTER)
	$(CC) -DMT -DNO_CACHE_COHERENCE -mos=rtems $(CFLAGS) $(SRC_MASTER) -o $(EXEC).master $(LIBS)

# Builds slave.
slave: $(SRC_SLAVE)
	$(CC) -DMT -DNO_CACHE_COHERENCE -mos=nodeos $(CFLAGS) $(SRC_SLAVE) -o $(EXEC).slave $(LIBS)
	
# Cleans compilation files.
clean:
	rm -f $(BINDIR)/$(EXEC).mppa.mpk