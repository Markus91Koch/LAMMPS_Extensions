/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <math.h>
#include <string.h>
#include "compute_asphericity_chunk.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "compute_chunk_atom.h"
#include "domain.h"
#include "math_extra.h"
#include "memory.h"
#include "error.h"


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeAsphericityChunk::ComputeAsphericityChunk(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  idchunk(NULL), massproc(NULL), masstotal(NULL), com(NULL), comall(NULL), 
  rg(NULL), rgall(NULL), rgt(NULL), rgtall(NULL), evalues(NULL),aspher(NULL)
{
  if (narg < 4) error->all(FLERR,"Illegal compute asphericity/chunk command");

  // ID of compute chunk/atom

  int n = strlen(arg[3]) + 1;
  idchunk = new char[n];
  strcpy(idchunk,arg[3]);

  init();

  // optional args

  tensor = 0;
  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"tensor") == 0) {
      tensor = 1;
      iarg++;
    } else error->all(FLERR,"Illegal compute asphericity/chunk command");
  }

  if (tensor) {
    array_flag = 1;
    size_array_cols = 3;
    size_array_rows = 0;
    size_array_rows_variable = 1;
    extarray = 0;
  } else {
    vector_flag = 1;
    size_vector = 0;
    size_vector_variable = 1;
    extvector = 0;
  }

  // chunk-based data

  nchunk = 1;
  maxchunk = 0;
  allocate();
}

/* ---------------------------------------------------------------------- */

ComputeAsphericityChunk::~ComputeAsphericityChunk()
{
  delete [] idchunk;
  memory->destroy(massproc);
  memory->destroy(masstotal);
  memory->destroy(com);
  memory->destroy(comall);
  memory->destroy(rg);
  memory->destroy(rgall);
  memory->destroy(rgt);
  memory->destroy(rgtall);
  memory->destroy(evalues);
  memory->destroy(aspher);  
}

/* ---------------------------------------------------------------------- */

void ComputeAsphericityChunk::init()
{
  int icompute = modify->find_compute(idchunk);
  if (icompute < 0)
    error->all(FLERR,"Chunk/atom compute does not exist for "
               "compute asphericity/chunk");
  cchunk = (ComputeChunkAtom *) modify->compute[icompute];
  if (strcmp(cchunk->style,"chunk/atom") != 0)
    error->all(FLERR,"Compute asphericity/chunk does not use chunk/atom compute");
}

/* ---------------------------------------------------------------------- */

void ComputeAsphericityChunk::compute_vector()
{
  int i,j,index;
  double dx,dy,dz,massone;
  double unwrap[3];
  double rgtensor[3][3];
  
  invoked_array = update->ntimestep;

  com_chunk();
  int *ichunk = cchunk->ichunk;

  for (i = 0; i < nchunk; i++)
    for (j = 0; j < 6; j++) rgt[i][j] = 0.0;

  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;
  imageint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      domain->unmap(x[i],image[i],unwrap);
      dx = unwrap[0] - comall[index][0];
      dy = unwrap[1] - comall[index][1];
      dz = unwrap[2] - comall[index][2];
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      rgt[index][0] += dx*dx * massone;
      rgt[index][1] += dy*dy * massone;
      rgt[index][2] += dz*dz * massone;
      rgt[index][3] += dx*dy * massone;
      rgt[index][4] += dx*dz * massone;
      rgt[index][5] += dy*dz * massone;
    }

  if (nchunk)
    MPI_Allreduce(&rgt[0][0],&rgtall[0][0],nchunk*6,MPI_DOUBLE,MPI_SUM,world);

  for (i = 0; i < nchunk; i++) {
    if (masstotal[i] > 0.0) {
      for (j = 0; j < 6; j++)
        rgtall[i][j] = rgtall[i][j]/masstotal[i];
    }
  }

  for (int m = 0; m < nchunk; m++) {
      rgtensor[0][0] = rgtall[m][0];
      rgtensor[1][1] = rgtall[m][1];
      rgtensor[2][2] = rgtall[m][2];
      rgtensor[0][1] = rgtall[m][3];
      rgtensor[0][2] = rgtall[m][4];
      rgtensor[2][1] = rgtall[m][5];
      rgtensor[1][0] = rgtall[m][3];
      rgtensor[2][0] = rgtall[m][4];
      rgtensor[1][2] = rgtall[m][5];

  double evectors[3][3], ev[3];
  
  if (MathExtra::jacobi(rgtensor,ev,evectors))
     error->all(FLERR,"Insufficient Jacobi rotations for rigid molecule");   
  
  //printf("B4 CHUNKNR %d max mid min %f %f %f\n", m, ev[0], ev[1],ev[2] );
  
  
  // Sorting of the eigenvalues (max, mid, min)
  double max, min, mid;
  int maxid, minid, midid;
  
  maxid=0;
  minid=0;
  midid=0;
  max = MAX(ev[0],ev[1]);
  min = MIN(ev[0],ev[1]);
  if (max>ev[maxid]) maxid=1;
  if (min<ev[minid]) minid=1;
  max = MAX(max,ev[2]);
  min = MIN(min,ev[2]);
  if (max>ev[maxid]) maxid=2;  
  if (min<ev[minid]) minid=2;
  
  if (midid==minid || midid==maxid) midid+=1;
  if (midid==minid || midid==maxid) midid+=1;
  mid=ev[midid];   
  
  //printf("AF CHUNKNR %d max mid min %f %f %f\n", m, max, mid,min );
  
  aspher[m]=max-0.5*(mid+min);
  
  
  }
    
}

/* ---------------------------------------------------------------------- */

void ComputeAsphericityChunk::compute_array()
{
  int i,j,index;
  double dx,dy,dz,massone;
  double unwrap[3];
  double rgtensor[3][3];
  
  invoked_array = update->ntimestep;

  com_chunk();
  int *ichunk = cchunk->ichunk;

  for (i = 0; i < nchunk; i++)
    for (j = 0; j < 6; j++) rgt[i][j] = 0.0;

  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;
  imageint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      domain->unmap(x[i],image[i],unwrap);
      dx = unwrap[0] - comall[index][0];
      dy = unwrap[1] - comall[index][1];
      dz = unwrap[2] - comall[index][2];
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      rgt[index][0] += dx*dx * massone;
      rgt[index][1] += dy*dy * massone;
      rgt[index][2] += dz*dz * massone;
      rgt[index][3] += dx*dy * massone;
      rgt[index][4] += dx*dz * massone;
      rgt[index][5] += dy*dz * massone;
    }

  if (nchunk)
    MPI_Allreduce(&rgt[0][0],&rgtall[0][0],nchunk*6,MPI_DOUBLE,MPI_SUM,world);

  for (i = 0; i < nchunk; i++) {
    if (masstotal[i] > 0.0) {
      for (j = 0; j < 6; j++)
        rgtall[i][j] = rgtall[i][j]/masstotal[i];
    }
  }

  
  for (int m = 0; m < nchunk; m++) {
      rgtensor[0][0] = rgtall[m][0];
      rgtensor[1][1] = rgtall[m][1];
      rgtensor[2][2] = rgtall[m][2];
      rgtensor[0][1] = rgtall[m][3];
      rgtensor[0][2] = rgtall[m][4];
      rgtensor[2][1] = rgtall[m][5];
      rgtensor[1][0] = rgtall[m][3];
      rgtensor[2][0] = rgtall[m][4];
      rgtensor[1][2] = rgtall[m][5];
  
  double evectors[3][3], ev[3];
  
  if (MathExtra::jacobi(rgtensor,ev,evectors))
     error->all(FLERR,"Insufficient Jacobi rotations for rigid molecule");   
  
  evalues[m][0]=ev[0];
  evalues[m][1]=ev[1];
  evalues[m][2]=ev[2];   
  }
  
}


/* ----------------------------------------------------------------------
   calculate per-chunk COM, used by both scalar and tensor
------------------------------------------------------------------------- */

void ComputeAsphericityChunk::com_chunk()
{
  int index;
  double massone;
  double unwrap[3];

  // compute chunk/atom assigns atoms to chunk IDs
  // extract ichunk index vector from compute
  // ichunk = 1 to Nchunk for included atoms, 0 for excluded atoms

  nchunk = cchunk->setup_chunks();
  cchunk->compute_ichunk();
  int *ichunk = cchunk->ichunk;

  if (nchunk > maxchunk) allocate();
  if (tensor) size_array_rows = nchunk;
  else size_vector = nchunk;

  // zero local per-chunk values

  for (int i = 0; i < nchunk; i++) {
    massproc[i] = 0.0;
    com[i][0] = com[i][1] = com[i][2] = 0.0;
  }

  // compute COM for each chunk

  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;
  imageint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      domain->unmap(x[i],image[i],unwrap);
      massproc[index] += massone;
      com[index][0] += unwrap[0] * massone;
      com[index][1] += unwrap[1] * massone;
      com[index][2] += unwrap[2] * massone;
    }

  MPI_Allreduce(massproc,masstotal,nchunk,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&com[0][0],&comall[0][0],3*nchunk,MPI_DOUBLE,MPI_SUM,world);

  for (int i = 0; i < nchunk; i++) {
    if (masstotal[i] > 0.0) {
      comall[i][0] /= masstotal[i];
      comall[i][1] /= masstotal[i];
      comall[i][2] /= masstotal[i];
    }
  }

  
  
}

/* ----------------------------------------------------------------------
   lock methods: called by fix ave/time
   these methods insure vector/array size is locked for Nfreq epoch
     by passing lock info along to compute chunk/atom
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   increment lock counter
------------------------------------------------------------------------- */

void ComputeAsphericityChunk::lock_enable()
{
  cchunk->lockcount++;
}

/* ----------------------------------------------------------------------
   decrement lock counter in compute chunk/atom, it if still exists
------------------------------------------------------------------------- */

void ComputeAsphericityChunk::lock_disable()
{
  int icompute = modify->find_compute(idchunk);
  if (icompute >= 0) {
    cchunk = (ComputeChunkAtom *) modify->compute[icompute];
    cchunk->lockcount--;
  }
}

/* ----------------------------------------------------------------------
   calculate and return # of chunks = length of vector/array
------------------------------------------------------------------------- */

int ComputeAsphericityChunk::lock_length()
{
  nchunk = cchunk->setup_chunks();
  return nchunk;
}

/* ----------------------------------------------------------------------
   set the lock from startstep to stopstep
------------------------------------------------------------------------- */

void ComputeAsphericityChunk::lock(Fix *fixptr, bigint startstep, bigint stopstep)
{
  cchunk->lock(fixptr,startstep,stopstep);
}

/* ----------------------------------------------------------------------
   unset the lock
------------------------------------------------------------------------- */

void ComputeAsphericityChunk::unlock(Fix *fixptr)
{
  cchunk->unlock(fixptr);
}

/* ----------------------------------------------------------------------
   free and reallocate per-chunk arrays
------------------------------------------------------------------------- */

void ComputeAsphericityChunk::allocate()
{
  memory->destroy(massproc);
  memory->destroy(masstotal);
  memory->destroy(com);
  memory->destroy(comall);
  memory->destroy(rg);
  memory->destroy(rgall);
  memory->destroy(rgt);
  memory->destroy(rgtall);
  memory->destroy(evalues);
  memory->destroy(aspher);
  maxchunk = nchunk;
  memory->create(massproc,maxchunk,"asphericity/chunk:massproc");
  memory->create(masstotal,maxchunk,"asphericity/chunk:masstotal");
  memory->create(com,maxchunk,3,"asphericity/chunk:com");
  memory->create(comall,maxchunk,3,"asphericity/chunk:comall");
    memory->create(rgt,maxchunk,6,"asphericity/chunk:rgt");
    memory->create(rgtall,maxchunk,6,"asphericity/chunk:rgtall");  
  if (tensor) {
    memory->create(evalues,maxchunk,3,"asphericity/chunk:evalues");
    array = evalues;
  } else {
    memory->create(aspher,maxchunk,"asphericity/chunk:aspher");
    vector = aspher;
  }
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputeAsphericityChunk::memory_usage()
{
  double bytes = (bigint) maxchunk * 2 * sizeof(double);
  bytes += (bigint) maxchunk * 2*3 * sizeof(double);
  if (tensor) bytes += (bigint) maxchunk * 2*3 * sizeof(double);
  else bytes += (bigint) maxchunk * 2 * sizeof(double);
  return bytes;
}
