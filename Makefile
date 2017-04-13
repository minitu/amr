DEFINE=-DTIMER #-DUSE_GPU #-DGPU_DEBUG
CHARMHOME ?= ../charm-cuda
CHARMC ?= $(CHARMHOME)/bin/charmc -I.
CXX=$(CHARMC)
CUDATOOLKIT_HOME ?= /usr/local/cuda
NVCC ?= $(CUDATOOLKIT_HOME)/bin/nvcc
NVCC_FLAGS = $(DEFINE) -c --std=c++11 #-O3 -use_fast_math
NVCC_INC = -I$(CUDATOOLKIT_HOME)/include -I$(CHARMHOME)/src/arch/cuda/hybridAPI -I./lib/cub-1.6.4
CHARMINC = -I$(CHARMHOME)/include
LD_LIBS = #-lcublas

OPTS ?= -O3 -g #-DUSE_GPU
CXXFLAGS += $(DEFINE) -DAMR_REVISION=$(REVNUM) $(OPTS)

OBJS = OctIndex.o Advection.o Main.o AdvectionCU.o

all: advection

advection: $(OBJS)
	$(CHARMC) $(CXXFLAGS) $(LDFLAGS) -language charm++ -o $@ $^ $(LD_LIBS) -module DistributedLB

Advection.decl.h Main.decl.h: advection.ci.stamp
advection.ci.stamp: advection.ci
	$(CHARMC) $<
	touch $@

Advection.o: Advection.C Advection.h OctIndex.h Main.decl.h Advection.decl.h
Main.o: Main.C Advection.h OctIndex.h Main.decl.h Advection.decl.h
OctIndex.o: OctIndex.C OctIndex.h Advection.decl.h
AdvectionCU.o: Advection.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) $(CHARMINC) -o AdvectionCU.o Advection.cu

test: all
	./charmrun +p8 ++local ./advection 3 32 30 9 +balancer DistributedLB

clean:
	rm -f *.decl.h *.def.h conv-host *.o advection charmrun advection.ci.stamp

bgtest: all
	./charmrun advection +p4 10 +x2 +y2 +z2
