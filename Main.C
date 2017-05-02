#include "Headers.h"
#include <assert.h>

using std::max;
using std::min;

/* readonly */ CProxy_Advection qtree;
/* readonly */ CProxy_Main mainProxy;

/* readonly */ int array_height;
/* readonly */ int array_width;
/* readonly */ int array_depth;

/* readonly */ int num_chare_rows;
/* readonly */ int num_chare_cols;
/* readonly */ int num_chare_Zs;

/* readonly */ int block_height;
/* readonly */ int block_width;
/* readonly */ int block_depth;

/* readonly */ int min_depth, max_depth;
/* readonly */ int max_iterations, refine_frequency;
/* readonly */ int lb_freq; // load balancing frequency

/* readonly */ const char amrRevision[] = INQUOTES(AMR_REVISION);

/* readonly */ float xmin, xmax, ymin, ymax, zmin, zmax;
/* readonly */ float xctr, yctr, zctr, radius;
/* readonly */ float dx, dy, dz, vx, vy, vz;
/* readonly */ float apx, anx, apy, any, apz, anz;
/* readonly */ float tmax, t, dt, cfl;

/* readonly */ float start_time, end_time;

Main::Main(CkArgMsg* m) {
  ckout << "* Running AMR code revision: " << amrRevision << endl;

  mainProxy = thisProxy;
  iterations = 0;

  // handle arguments
  if(m->argc < 5 || m->argc > 6) {
    ckout << "Usage: " << m->argv[0] << " [max_depth] [block_size] [iterations] [lb_freq] [array_dim]?" << endl; 
    CkExit();
  }

  // set max depth
  max_depth = atoi(m->argv[1]);
  if (max_depth >= 11)
    ckout << "Depth too high for bitvector index" << endl;

  // set block size
  block_width = block_height = block_depth = atoi(m->argv[2]);

  // set number of iterations
  max_iterations = atoi(m->argv[3]);

  // set load balancing frequency
  lb_freq = atoi(m->argv[4]);
  refine_frequency = 3;
  if (lb_freq % refine_frequency != 0) {
    ckout << "Load balancing frequency should be a mulitple of refine frequency (3)" << endl;
    CkExit();
  }

  // set entire grid size
  if (m->argc == 6)
    array_width = array_height = array_depth = atoi(m->argv[5]);
  else
    array_width = array_height = array_depth = 128;

  if(array_width < block_width || array_width % block_width < 0) {
    ckout << "Incompatible arguments: array size = " << array_width << "block size = " << block_width << endl;
    CkExit();
  }

  // set number of chares per dimension
  num_chare_rows = num_chare_cols = num_chare_Zs = array_width/block_width;
  num_chares = num_chare_rows * num_chare_cols * num_chare_Zs;

  // set min depth
  float fdepth = log(num_chares) / log(NUM_CHILDREN);
  min_depth = (fabs(fdepth - ceil(fdepth)) < 0.000001) ? ceil(fdepth) : floor(fdepth);
  if (min_depth == 0)
    min_depth = 1; // should be at least 1 in any case

  // initialize constants
  xmin = 0;
  xmax = 1;
  ymin = 0;
  ymax = 1;
  zmin = 0;
  zmax = 1;
  t = 0;
  tmax = 10000;
  cfl = 0.4;
  vx = 0.0;
  vy = 0.0;
  vz = 0.1;

  dx = (xmax - xmin)/float(array_width);
  dy = (ymax - ymin)/float(array_height);
  dz = (zmax - zmin)/float(array_depth);

  xctr = 0.5;
  yctr = 0.5;
  zctr = 0.5;

  radius = 0.2;

  apx = max(vx, (float)0.0);
  anx = min(vx, (float)0.0);
  apy = max(vy, (float)0.0);
  any = min(vy, (float)0.0);
  apz = max(vz, (float)0.0);
  anz = min(vz, (float)0.0);

  // create tree of chares
  CProxy_AdvMap map = CProxy_AdvMap::ckNew();
  CkArrayOptions opts;
  opts.setMap(map);
  qtree = CProxy_Advection::ckNew(opts);

  dt = min(min(dx,dy),dz) / sqrt(vx*vx + vy*vy + vz*vz) * cfl;
  dt /= pow(2., max_depth - min_depth);
  if ((t + dt) >= tmax)
    dt = tmax - t;
  t = t + dt;

  CkPrintf("* Constants\n\tdx = %f, apx = %f, anx = %f, dt = %f, t = %f\n", dx, apx, anx, dt, t);
  CkPrintf("* Running Advection on %d processor(s)\n"
      "\tArray dimension: %d x %d x %d\n"
      "\tBlock dimension: %d x %d x %d\n"
      "\tNumber of chares: %d x %d x %d\n"
      "\tMinimum depth: %d\n"
      "\tMaximum depth: %d\n"
      "\tMaximum number of iterations: %d\n"
      "\tLoad balacning frequency: %d\n",
      CkNumPes(), array_width, array_height, array_depth,
      block_width, block_height, block_depth,
      num_chare_rows, num_chare_cols, num_chare_Zs,
      min_depth, max_depth, max_iterations, lb_freq);

  // dynamic insertion of chares
  for (int i = 0; i < num_chare_rows; ++i)
    for (int j = 0; j < num_chare_cols; ++j)
      for (int k = 0; k < num_chare_Zs; ++k)
        qtree[OctIndex(i, j, k, min_depth)].insert(xmin, xmax, ymin, ymax, zmin, zmax);
  qtree.doneInserting();

  // begin simulation
  CkStartQD(CkCallback(CkIndex_Main::startMeshGeneration(), thisProxy));
  ppc = CProxy_AdvectionGroup::ckNew();
}

void Main::startMeshGeneration() {
  start_time = CkWallTimer();
  qtree.iterate();
}

void Main::terminate(){
  ckout << "simulation time: " << CkWallTimer() - start_time << " s" << endl;
  ppc.reduceWorkUnits();
}

void Main::totalWorkUnits(int total) {
  CkPrintf("total work units = %d\n", total);
  nresponses = 0;
  ppc.reduceQdTimes();
}

#define GOLDEN_RATIO_PRIME_64 0x9e37fffffffc0001ULL

struct AdvMap : public CBase_AdvMap {
  int bits;
  AdvMap() : bits(log2(CkNumPes())) { }
  
  void pup(PUP::er &p){ bits = log2(CkNumPes()); }
  AdvMap(CkMigrateMessage *m): CBase_AdvMap(m),bits(log2(CkNumPes())){}

  int procNum(int arrayHdl, const CkArrayIndex& i) {
    int numPes = CkNumPes();
    const OctIndex& idx = *reinterpret_cast<const OctIndex*>(i.data());
    int baseBits = 8;

    unsigned long long val = idx.bitVector >> (sizeof(unsigned int)*8 - baseBits);
    unsigned long long hash = GOLDEN_RATIO_PRIME_64 * val;

    int basePE = hash >> (64 - bits);

    unsigned long validBits = idx.bitVector & ((1L << 24) - 1);
    validBits += (1L << 22);
    unsigned long offset = validBits >> (sizeof(unsigned int)*8 - idx.nbits);
    offset += (idx.nbits == 8);

    int pe = (basePE + offset - 1) % numPes;

    return pe;
  }
};

#include "Main.def.h"
