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
#include "compute_ellipsoid_chunk.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "compute_chunk_atom.h"
#include "domain.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "variable.h"
#include "input.h"

using namespace LAMMPS_NS;
enum{PROLATE,OBLATE};
// PROLATE = 0
// OBLATE = 1
enum{NONE,CONSTANT,EQUAL,ATOM};
// NONE = 0
// CONSTANT = 1
// EQUAL = 2
// ATOM = 3
#define EPSILON 1.0e-6

/* ---------------------------------------------------------------------- */

ComputeEllipsoidChunk::ComputeEllipsoidChunk(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  idchunk(NULL), massproc(NULL), masstotal(NULL), com(NULL), comall(NULL), 
  rg(NULL), rgall(NULL), rgt(NULL), rgtall(NULL), sorted_evec(NULL),scalarprod(NULL),
  xstr(NULL), ystr(NULL), zstr(NULL)
{
  if (narg < 8) error->all(FLERR,"Illegal compute ellipsoid/chunk command");

  
  // input vector (gives input vector)
  
    if (strstr(arg[3],"v_") == arg[4]) {
    int n = strlen(&arg[3][2]) + 1;
    xstr = new char[n];
    strcpy(xstr,&arg[3][2]);
  } else {
    xvalue = force->numeric(FLERR,arg[3]);
    xstyle = CONSTANT;
  }
  if (strstr(arg[4],"v_") == arg[4]) {
    int n = strlen(&arg[4][2]) + 1;
    ystr = new char[n];
    strcpy(ystr,&arg[4][2]);
  } else {
    yvalue = force->numeric(FLERR,arg[4]);
    ystyle = CONSTANT;
  }
  if (strstr(arg[5],"v_") == arg[5]) {
    int n = strlen(&arg[5][2]) + 1;
    zstr = new char[n];
    strcpy(zstr,&arg[5][2]);
  } else {
    zvalue = force->numeric(FLERR,arg[5]);
    zstyle = CONSTANT;
  }  

  
  // ID of compute chunk/atom

  int n = strlen(arg[6]) + 1;
  idchunk = new char[n];
  strcpy(idchunk,arg[6]);

  init();
  
   // mesogen type - prolate (stick) or oblate (disk)
  if (strcmp(arg[7],"prolate") == 0) {
    which = PROLATE;
  } else if (strcmp(arg[7],"oblate") == 0) {
    which = OBLATE;
  }
  else {
    error->all(FLERR,"Illegal compute ellipsoid/chunk command");
  }  
  
  
  // optional args - output as tensor or vector

  tensor = 0;
  int iarg = 8;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"tensor") == 0) {
      tensor = 1;
      iarg++;
    } else error->all(FLERR,"Illegal compute ellipsoid/chunk command");
  }

  if (tensor) {
    array_flag = 1;
    size_array_cols = 12;
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

ComputeEllipsoidChunk::~ComputeEllipsoidChunk()
{
  delete [] idchunk;
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
  memory->destroy(massproc);
  memory->destroy(masstotal);
  memory->destroy(com);
  memory->destroy(comall);
  memory->destroy(rg);
  memory->destroy(rgall);
  memory->destroy(rgt);
  memory->destroy(rgtall);
  memory->destroy(sorted_evec);
  memory->destroy(scalarprod);
  
}

/* ---------------------------------------------------------------------- */

void ComputeEllipsoidChunk::init()
{
  int icompute = modify->find_compute(idchunk);
  if (icompute < 0)
    error->all(FLERR,"Chunk/atom compute does not exist for "
               "compute ellipsoid/chunk");
  cchunk = (ComputeChunkAtom *) modify->compute[icompute];
  if (strcmp(cchunk->style,"chunk/atom") != 0)
    error->all(FLERR,"Compute ellipsoid/chunk does not use chunk/atom compute");
  
   // check variables

  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0)
      error->all(FLERR,"x variable name for compute ellipsoid/chunk does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else error->all(FLERR,"x variable for compute ellipsoid/chunk is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0)
      error->all(FLERR,"y variable name for compute ellipsoid/chunk does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else error->all(FLERR,"y variable for compute ellipsoid/chunk is invalid style");
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0)
      error->all(FLERR,"z variable name for compute ellipsoid/chunk does not exist");
    if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
    else error->all(FLERR,"z variable for compute ellipsoid/chunk is invalid style");
  }

  //if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL)
   // varflag = EQUAL;
  //else varflag = CONSTANT;
  
}

/* ---------------------------------------------------------------------- */

void ComputeEllipsoidChunk::compute_vector()
{
  int i,j,index;
  double dx,dy,dz,massone;
  double unwrap[3];
  double rgtensor[3][3];
  double ex[3], ey[3], ez[3], cross[3];
  double invec[3];
  double rmaxev[3];
  double rgex[3],rgey[3],rgez[3];
  invoked_array = update->ntimestep;

  //invec[0]=0.0;
  //invec[1]=0.0;  
  //invec[2]=1.0;  
  
  if (varflag == EQUAL) {
    modify->clearstep_compute();
    if (xstyle == EQUAL) xvalue = input->variable->compute_equal(xvar);
    if (ystyle == EQUAL) yvalue = input->variable->compute_equal(yvar);
    if (zstyle == EQUAL) zvalue = input->variable->compute_equal(zvar);
    modify->addstep_compute(update->ntimestep + 1);
  }


 //normalize input vector
  invec[0]=xvalue;
  invec[1]=yvalue;
  invec[2]=zvalue;
  
  if (invec[0] == 0.0 && invec[1] == 0.0 && invec[2] == 0.0)
    error->all(FLERR,"Invalid input vector");
  MathExtra::norm3(invec); // normalize vector to 1  
  
  
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
  
  rgex[0] = evectors[0][0];
  rgex[1] = evectors[1][0];
  rgex[2] = evectors[2][0];
  rgey[0] = evectors[0][1];
  rgey[1] = evectors[1][1];
  rgey[2] = evectors[2][1];
  rgez[0] = evectors[0][2];
  rgez[1] = evectors[1][2];
  rgez[2] = evectors[2][2];    
  
/*   ex[0]=rgex[0];
   ex[1]=rgex[1];
   ex[2]=rgex[2];
   ey[0]=rgey[0];
   ey[1]=rgey[1];
   ey[2]=rgey[2];
   ez[0]=rgez[0];
   ez[1]=rgez[1];
   ez[2]=rgez[2];*/   
  
    MathExtra::cross3(rgex,rgey,cross);
  if (MathExtra::dot3(cross,rgez) < 0.0){
      printf("flipp\n");
      MathExtra::negate3(rgez);
/*      rgez[m][0]=ez[0];
      rgez[m][1]=ez[1];
      rgez[m][2]=ez[2];*/      
      
  }
  
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
  
  if (ev[0] < EPSILON*max) ev[0] = 0.0;
  if (ev[1] < EPSILON*max) ev[1] = 0.0;
  if (ev[2] < EPSILON*max) ev[2] = 0.0;  

//   printf("b4 chunk %d max %f mid %f min %f\n", m, max, mid, min);
  
  max=ev[maxid];
  mid=ev[midid];
  min=ev[minid];
//   printf("after chunk %d max %f mid %f min %f\n", m, max, mid, min);
    
  

   
   //double cosine, cosine2;
   
   //scalarprod[m]=rmaxev[0]*invec[0]+rmaxev[1]*invec[1]+rmaxev[2]*invec[2];
   //cosine       =rmaxev[0]*invec[0]+rmaxev[1]*invec[1]+rmaxev[2]*invec[2];
   //cosine2=cosine*cosine;    
   //should be called theta
   //scalarprod[m]=acos(cosine)*180/MathConst::MY_PI;
   
   // if prolate take evector or biggest eigenvalue and 
  // if oblate take evector of smallest eigenvalue
    if (which == PROLATE) {
           rmaxev[0]=evectors[0][maxid];
           rmaxev[1]=evectors[1][maxid];
           rmaxev[2]=evectors[2][maxid]; 
    }
    else if (which == OBLATE) {
           rmaxev[0]=evectors[0][minid];
           rmaxev[1]=evectors[1][minid];
           rmaxev[2]=evectors[2][minid]; 
    }
     MathExtra::norm3(rmaxev);
    
     double cosine;  // , cosine2
     cosine=rmaxev[0]*invec[0]+rmaxev[1]*invec[1]+rmaxev[2]*invec[2];
     scalarprod[m]=cosine*cosine;  // actually print out cosine squared !!!

     // scalarprod[m]=rmaxev[0]*invec[0]+rmaxev[1]*invec[1]+rmaxev[2]*invec[2];
     
    
//   rminev[m][0]=evectors[0][minid];
//   rminev[m][1]=evectors[1][minid];
//   rminev[m][2]=evectors[2][minid];
//   rmidev[m][0]=evectors[0][midid];
//   rmidev[m][1]=evectors[1][midid];
//   rmidev[m][2]=evectors[2][midid];
  
  //evalues[m][0]=ev[0];
  //evalues[m][1]=ev[1];
  //evalues[m][2]=ev[2];
//   sorted_evec[m][0]=evectors[0][maxid];
//   sorted_evec[m][1]=evectors[1][maxid];
//   sorted_evec[m][2]=evectors[2][maxid];
//   
//   sorted_evec[m][3]=evectors[0][midid];
//   sorted_evec[m][4]=evectors[1][midid];
//   sorted_evec[m][5]=evectors[2][midid];
//   
//   sorted_evec[m][6]=evectors[0][minid];
//   sorted_evec[m][7]=evectors[1][minid];
//   sorted_evec[m][8]=evectors[2][minid];
//   
  
  
  
  
  }
    
}

/* ---------------------------------------------------------------------- */

void ComputeEllipsoidChunk::compute_array()
{
  int i,j,index;
  double dx,dy,dz,massone;
  double unwrap[3];
  double rgtensor[3][3];
  double ex[3], ey[3], ez[3], cross[3];
  double rgex[3],rgey[3],rgez[3];
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
  
  rgex[0] = evectors[0][0];
  rgex[1] = evectors[1][0];
  rgex[2] = evectors[2][0];
  rgey[0] = evectors[0][1];
  rgey[1] = evectors[1][1];
  rgey[2] = evectors[2][1];
  rgez[0] = evectors[0][2];
  rgez[1] = evectors[1][2];
  rgez[2] = evectors[2][2];    
  
/*   ex[0]=rgex[0];
   ex[1]=rgex[1];
   ex[2]=rgex[2];
   ey[0]=rgey[0];
   ey[1]=rgey[1];
   ey[2]=rgey[2];
   ez[0]=rgez[0];
   ez[1]=rgez[1];
   ez[2]=rgez[2];*/   
  
    MathExtra::cross3(rgex,rgey,cross);
  if (MathExtra::dot3(cross,rgez) < 0.0){
      printf("flipp\n");
      MathExtra::negate3(rgez);
/*      rgez[m][0]=ez[0];
      rgez[m][1]=ez[1];
      rgez[m][2]=ez[2];*/      
      
  }
  
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
  
  if (ev[0] < EPSILON*max) ev[0] = 0.0;
  if (ev[1] < EPSILON*max) ev[1] = 0.0;
  if (ev[2] < EPSILON*max) ev[2] = 0.0;  

//   printf("b4 chunk %d max %f mid %f min %f\n", m, max, mid, min);
  
  max=ev[maxid];
  mid=ev[midid];
  min=ev[minid];
//   printf("after chunk %d max %f mid %f min %f\n", m, max, mid, min);
    
  
//   rmaxev[m][0]=evectors[0][maxid];
//   rmaxev[m][1]=evectors[1][maxid];
//   rmaxev[m][2]=evectors[2][maxid];  
//   rminev[m][0]=evectors[0][minid];
//   rminev[m][1]=evectors[1][minid];
//   rminev[m][2]=evectors[2][minid];
//   rmidev[m][0]=evectors[0][midid];
//   rmidev[m][1]=evectors[1][midid];
//   rmidev[m][2]=evectors[2][midid];
  
  //evalues[m][0]=ev[0];
  //evalues[m][1]=ev[1];
  //evalues[m][2]=ev[2];
  sorted_evec[m][0]=evectors[0][maxid];
  sorted_evec[m][1]=evectors[1][maxid];
  sorted_evec[m][2]=evectors[2][maxid];
  
  sorted_evec[m][3]=evectors[0][midid];
  sorted_evec[m][4]=evectors[1][midid];
  sorted_evec[m][5]=evectors[2][midid];
  
  sorted_evec[m][6]=evectors[0][minid];
  sorted_evec[m][7]=evectors[1][minid];
  sorted_evec[m][8]=evectors[2][minid];
  
  sorted_evec[m][9]=max;
  sorted_evec[m][10]=mid;
  sorted_evec[m][11]=min;  
  
  }
  
}


/* ----------------------------------------------------------------------
   calculate per-chunk COM, used by both scalar and tensor
------------------------------------------------------------------------- */

void ComputeEllipsoidChunk::com_chunk()
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

void ComputeEllipsoidChunk::lock_enable()
{
  cchunk->lockcount++;
}

/* ----------------------------------------------------------------------
   decrement lock counter in compute chunk/atom, it if still exists
------------------------------------------------------------------------- */

void ComputeEllipsoidChunk::lock_disable()
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

int ComputeEllipsoidChunk::lock_length()
{
  nchunk = cchunk->setup_chunks();
  return nchunk;
}

/* ----------------------------------------------------------------------
   set the lock from startstep to stopstep
------------------------------------------------------------------------- */

void ComputeEllipsoidChunk::lock(Fix *fixptr, bigint startstep, bigint stopstep)
{
  cchunk->lock(fixptr,startstep,stopstep);
}

/* ----------------------------------------------------------------------
   unset the lock
------------------------------------------------------------------------- */

void ComputeEllipsoidChunk::unlock(Fix *fixptr)
{
  cchunk->unlock(fixptr);
}

/* ----------------------------------------------------------------------
   free and reallocate per-chunk arrays
------------------------------------------------------------------------- */

void ComputeEllipsoidChunk::allocate()
{
  memory->destroy(massproc);
  memory->destroy(masstotal);
  memory->destroy(com);
  memory->destroy(comall);
  memory->destroy(rg);
  memory->destroy(rgall);
  memory->destroy(rgt);
  memory->destroy(rgtall);
  memory->destroy(sorted_evec);
  memory->destroy(scalarprod);
  maxchunk = nchunk;
  memory->create(massproc,maxchunk,"ellipsoid/chunk:massproc");
  memory->create(masstotal,maxchunk,"ellipsoid/chunk:masstotal");
  memory->create(com,maxchunk,3,"ellipsoid/chunk:com");
  memory->create(comall,maxchunk,3,"ellipsoid/chunk:comall");
    memory->create(rgt,maxchunk,6,"ellipsoid/chunk:rgt");
    memory->create(rgtall,maxchunk,6,"ellipsoid/chunk:rgtall");  
  if (tensor) {
    memory->create(sorted_evec,maxchunk,12,"ellipsoid/chunk:sorted_evec");
    array = sorted_evec;
  } else {
    memory->create(scalarprod,maxchunk,"ellipsoid/chunk:scalarprod");
    vector = scalarprod;
  }
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputeEllipsoidChunk::memory_usage()
{
  double bytes = (bigint) maxchunk * 2 * sizeof(double);
  bytes += (bigint) maxchunk * 2*3 * sizeof(double);
  if (tensor) bytes += (bigint) maxchunk * 2*12 * sizeof(double);
  else bytes += (bigint) maxchunk * 2 * sizeof(double);
  return bytes;
}
