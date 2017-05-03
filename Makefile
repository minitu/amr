CHARM_HOME = ../charm-cuda
DEFINE= -DTIMER # Possible flags: -DTIMER

CHARMC = $(CHARM_HOME)/bin/charmc -I.
CXX=$(CHARMC)
OPTS = -O0 -g
CXXFLAGS += $(DEFINE) -DAMR_REVISION=$(REVNUM) $(OPTS)
LD_LIBS =
CUDA_LD_LIBS = -L$(CUDATOOLKIT_HOME)/lib64 -lcudart
OBJS = OctIndex.o Advection.o Main.o

CUDATOOLKIT_HOME ?= /usr/local/cuda
NVCC ?= $(CUDATOOLKIT_HOME)/bin/nvcc
NVCC_FLAGS = -c --std=c++11 -O3
NVCC_INC = -I$(CUDATOOLKIT_HOME)/include -I$(CHARM_HOME)/src/arch/cuda/hybridAPI -I./lib/cub-1.6.4
CHARMINC = -I$(CHARM_HOME)/include
GPU_OBJS = OctIndex.o AdvectionGPU.o Main.o AdvectionCU.o
GPUMANAGER_OBJS = OctIndex.o AdvectionGPUManager.o Main.o AdvectionCUGPUManager.o

all: advection cuda

advection: $(OBJS)
	$(CHARMC) $(CXXFLAGS) $(LDFLAGS) -language charm++ -o $@ $^ $(LD_LIBS) -module DistributedLB

cuda: $(GPU_OBJS)
	$(CHARMC) $(CXXFLAGS) $(LDFLAGS) -language charm++ -o advection-$@ $^ $(LD_LIBS) $(CUDA_LD_LIBS) -module DistributedLB

gpumanager: $(GPUMANAGER_OBJS)
	$(CHARMC) $(CXXFLAGS) $(LDFLAGS) -language charm++ -o advection-$@ $^ $(LD_LIBS) -module DistributedLB

Advection.decl.h Main.decl.h: advection.ci.stamp
advection.ci.stamp: advection.ci
	$(CHARMC) $<
	touch $@

Advection.o: Advection.C Advection.h OctIndex.h Main.decl.h Advection.decl.h
Main.o: Main.C Advection.h OctIndex.h Main.decl.h Advection.decl.h
OctIndex.o: OctIndex.C OctIndex.h Advection.decl.h

AdvectionGPU.o: Advection.C Advection.h OctIndex.h Main.decl.h Advection.decl.h
	$(CHARMC) $(CXXFLAGS) -DUSE_GPU -c $< -o $@
AdvectionCU.o: Advection.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) $(CHARMINC) -o AdvectionCU.o Advection.cu

AdvectionGPUManager.o: Advection.C Advection.h OctIndex.h Main.decl.h Advection.decl.h
	$(CHARMC) $(CXXFLAGS) -DUSE_GPU -DUSE_GPUMANAGER -c $< -o $@
AdvectionCUGPUManager.o: Advection.cu
	$(NVCC) $(NVCC_FLAGS) -DGPU_MEMPOOL -DUSE_GPUMANAGER $(NVCC_INC) $(CHARMINC) -o $@ $<

test: advection
	./charmrun +p8 ++local ./$< 3 32 30 9 +balancer DistributedLB

test-cuda: advection-cuda
	./charmrun +p8 ++local ./$< 3 32 30 9 +balancer DistributedLB

test-gpumanager: advection-gpumanager
	./charmrun +p8 ++local ./$< 3 32 30 9 +balancer DistributedLB

clean:
	rm -f *.decl.h *.def.h conv-host *.o advection advection-cuda advection-gpumanager charmrun advection.ci.stamp

bgtest: advection
	./charmrun advection +p4 10 +x2 +y2 +z2
