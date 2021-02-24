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
#include "compute_nematic_chunk.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "compute_chunk_atom.h"
#include "domain.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

enum{PROLATE,OBLATE};

#define EPSILON 1.0e-6

/* ---------------------------------------------------------------------- */

ComputeNematicChunk::ComputeNematicChunk(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  idchunk(NULL), massproc(NULL), masstotal(NULL), com(NULL), comall(NULL), 
  rg(NULL), rgall(NULL), rgt(NULL), rgtall(NULL), Q(NULL)
{
  if (narg != 5) error->all(FLERR,"Illegal compute nematic/chunk command");

  // ID of compute chunk/atom

  int n = strlen(arg[3]) + 1;
  idchunk = new char[n];
  strcpy(idchunk,arg[3]);


  // mesogen type
  if (strcmp(arg[4],"prolate") == 0) {
    which = PROLATE;
  } else if (strcmp(arg[4],"oblate") == 0) {
    which = OBLATE;
  }
  else {
    error->all(FLERR,"Illegal compute chunk/atom command");
  }
  int iarg = 4;
  
  init();  
//   // optional args
// 
//   //tensor = 0;
//   int iarg = 4;
//   while (iarg < narg) {
//     if (strcmp(arg[iarg],"tensor") == 0) {
//       tensor = 1;
//       iarg++;
//     } else error->all(FLERR,"Illegal compute nematic/chunk command");
//   }

    vector_flag = 1;
    scalar_flag = 1;
    size_vector = 12;
    extscalar = 0;
    extvector = 0;  

  // chunk-based data

  nchunk = 1;
  maxchunk = 0;
  allocate();
  
  vector = new double[12];  
}

/* ---------------------------------------------------------------------- */

ComputeNematicChunk::~ComputeNematicChunk()
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
  memory->destroy(Q); 
}

/* ---------------------------------------------------------------------- */

void ComputeNematicChunk::init()
{
  int icompute = modify->find_compute(idchunk);
  if (icompute < 0)
    error->all(FLERR,"Chunk/atom compute does not exist for "
               "compute nematic/chunk");
  cchunk = (ComputeChunkAtom *) modify->compute[icompute];
  if (strcmp(cchunk->style,"chunk/atom") != 0)
    error->all(FLERR,"Compute nematic/chunk does not use chunk/atom compute");
}

/* ---------------------------------------------------------------------- */
// CALCULATE NEMATIC ORDER PARAMETER P2                
/* ---------------------------------------------------------------------- */

double ComputeNematicChunk::compute_scalar()
{
  int i,j,index;
  double dx,dy,dz,massone;
  double unwrap[3];
  double rgtensor[3][3];
  double Qtensor[3][3];
  double ex[3], ey[3], ez[3], cross[3];
  double invec[3];
  double rmaxev[3];
  double rgex[3],rgey[3],rgez[3];
  invoked_array = update->ntimestep;

  invec[0]=0.0;
  invec[1]=0.0;  
  invec[2]=1.0;  
  
  com_chunk();
  int *ichunk = cchunk->ichunk;

  for (i = 0; i < nchunk; i++){
    for (j = 0; j < 6; j++){
        rgt[i][j] = 0.0;
    }
  }
  
  for (j = 0; j < 6; j++){
        Q[j] = 0.0;
  }  

  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;
  imageint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  // calculate gyration tensor
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

      // diagonalize gyration tensor
  double evectors[3][3], ev[3];
  
  if (MathExtra::jacobi(rgtensor,ev,evectors))
     error->all(FLERR,"RG: Insufficient Jacobi rotations for rigid molecule");   
  
  // store eigenvectors for the moment
  
  rgex[0] = evectors[0][0];
  rgex[1] = evectors[1][0];
  rgex[2] = evectors[2][0];
  rgey[0] = evectors[0][1];
  rgey[1] = evectors[1][1];
  rgey[2] = evectors[2][1];
  rgez[0] = evectors[0][2];
  rgez[1] = evectors[1][2];
  rgez[2] = evectors[2][2];    
  
    MathExtra::cross3(rgex,rgey,cross);
  if (MathExtra::dot3(cross,rgez) < 0.0){
      printf("flipp\n");
      MathExtra::negate3(rgez);
/*      rgez[m][0]=ez[0];
      rgez[m][1]=ez[1];
      rgez[m][2]=ez[2];*/      
  }

  // order by largest / smallest eigenvector
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

//   if (ev[0] < EPSILON*max) ev[0] = 0.0;
//   if (ev[1] < EPSILON*max) ev[1] = 0.0;
//   if (ev[2] < EPSILON*max) ev[2] = 0.0;
// 
//   max=ev[maxid];
//   mid=ev[midid];
//   min=ev[minid];
  
  
  
  // define "maxis" = molecular axis
  // for prolate take the longest axis
  // for oblate take the shortest axis
  
  double maxis[3];  
  
  if (which == PROLATE) {
    maxis[0]=evectors[0][maxid];
    maxis[1]=evectors[1][maxid];
    maxis[2]=evectors[2][maxid];    
  }
  else if (which == OBLATE) {
    maxis[0]=evectors[0][minid];
    maxis[1]=evectors[1][minid];
    maxis[2]=evectors[2][minid];    
  }
  
  // nematic tensor  
  
  Q[0]+=3.0*maxis[0]*maxis[0]-1.0;
  Q[1]+=3.0*maxis[1]*maxis[1]-1.0;
  Q[2]+=3.0*maxis[2]*maxis[2]-1.0;
  Q[3]+=3.0*maxis[0]*maxis[1];
  Q[4]+=3.0*maxis[0]*maxis[2];
  Q[5]+=3.0*maxis[1]*maxis[2];
  
  }

  Q[0]/=2.0*nchunk;
  Q[1]/=2.0*nchunk;
  Q[2]/=2.0*nchunk;
  Q[3]/=2.0*nchunk;
  Q[4]/=2.0*nchunk;
  Q[5]/=2.0*nchunk;
  
  //diagonalize Q
  Qtensor[0][0]=Q[0];
  Qtensor[1][1]=Q[1];
  Qtensor[2][2]=Q[2];
  Qtensor[0][1]=Q[3];
  Qtensor[1][0]=Q[3];
  Qtensor[0][2]=Q[4];
  Qtensor[2][0]=Q[4];
  Qtensor[1][2]=Q[5];
  Qtensor[2][1]=Q[5];  

  double Qvectors[3][3], Qval[3];
  
  if (MathExtra::jacobi(Qtensor,Qval,Qvectors))
     error->all(FLERR,"Q: Insufficient Jacobi rotations for rigid molecule");   

  // order by largest / smallest eigenvector
  double qmax, qmin, qmid;
  int qmaxid, qminid, qmidid;
  double Qmaxev[3];

//  qmaxid=0;
//  qminid=0;
//  qmidid=0;
//  qmax = MAX(Qval[0],Qval[1]);
//  qmin = MIN(Qval[0],Qval[1]);
//  if (qmax>Qval[qmaxid]) qmaxid=1;
//  if (qmin<Qval[qminid]) qminid=1;
//  qmax = MAX(qmax,Qval[2]);
//  qmin = MIN(qmin,Qval[2]);
//  if (qmax>Qval[qmaxid]) qmaxid=2;
//  if (qmin<Qval[qminid]) qminid=2;
//
//  if (qmidid==qminid || qmidid==qmaxid) qmidid+=1;
//  if (qmidid==qminid || qmidid==qmaxid) qmidid+=1;
//  qmid=Qval[qmidid];
//
//  if (Qval[0] < EPSILON*qmax) Qval[0] = 0.0;
//  if (Qval[1] < EPSILON*qmax) Qval[1] = 0.0;
//  if (Qval[2] < EPSILON*qmax) Qval[2] = 0.0;

  //qmax=Qval[qmaxid];
  //qmid=Qval[qmidid];
  //qmin=Qval[qminid];  

  qmaxid=0;
  qminid=0;
  qmidid=0; // CAUTION!! SORTING OF ABSOLUTE VALUES, FOR SIGNED VALUES USE INDICES LATER
  qmax = MAX(fabs(Qval[0]),fabs(Qval[1]));
  qmin = MIN(fabs(Qval[0]),fabs(Qval[1]));
  if (qmax>fabs(Qval[qmaxid])) qmaxid=1;
  if (qmin<fabs(Qval[qminid])) qminid=1;
  qmax = MAX(qmax,fabs(Qval[2]));
  qmin = MIN(qmin,fabs(Qval[2]));
  if (qmax>fabs(Qval[qmaxid])) qmaxid=2;
  if (qmin<fabs(Qval[qminid])) qminid=2;

  if (qmidid==qminid || qmidid==qmaxid) qmidid+=1;
  if (qmidid==qminid || qmidid==qmaxid) qmidid+=1;
  qmid=fabs(Qval[qmidid]);


  scalar = Qval[qmaxid];
  return scalar; 
}

/* ---------------------------------------------------------------------- */
// CALCULATE DIRECTOR FOR THE SYSTEM
/* ---------------------------------------------------------------------- */

void ComputeNematicChunk::compute_vector()
{
  int i,j,index;
  double dx,dy,dz,massone;
  double unwrap[3];
  double rgtensor[3][3];
  double Qtensor[3][3];
  double ex[3], ey[3], ez[3], cross[3];
  double invec[3];
  double rmaxev[3];
  double rgex[3],rgey[3],rgez[3];
  invoked_array = update->ntimestep;

  invec[0]=0.0;
  invec[1]=0.0;  
  invec[2]=1.0;  
  
  com_chunk();
  int *ichunk = cchunk->ichunk;

  for (i = 0; i < nchunk; i++){
    for (j = 0; j < 6; j++){
        rgt[i][j] = 0.0;
    }
  }
  
  for (j = 0; j < 6; j++){
        Q[j] = 0.0;
  }  

  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;
  imageint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  // calculate gyration tensor
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

      // diagonalize gyration tensor
  double evectors[3][3], ev[3];
  
  if (MathExtra::jacobi(rgtensor,ev,evectors))
     error->all(FLERR,"RG: Insufficient Jacobi rotations for rigid molecule");   
  
  // store eigenvectors for the moment
  
  rgex[0] = evectors[0][0];
  rgex[1] = evectors[1][0];
  rgex[2] = evectors[2][0];
  rgey[0] = evectors[0][1];
  rgey[1] = evectors[1][1];
  rgey[2] = evectors[2][1];
  rgez[0] = evectors[0][2];
  rgez[1] = evectors[1][2];
  rgez[2] = evectors[2][2];    
  
    MathExtra::cross3(rgex,rgey,cross);
  if (MathExtra::dot3(cross,rgez) < 0.0){
      printf("flipp\n");
      MathExtra::negate3(rgez);
/*      rgez[m][0]=ez[0];
      rgez[m][1]=ez[1];
      rgez[m][2]=ez[2];*/      
  }

  // order by largest / smallest eigenvector
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
/*
  if (ev[0] < EPSILON*max) ev[0] = 0.0;
  if (ev[1] < EPSILON*max) ev[1] = 0.0;
  if (ev[2] < EPSILON*max) ev[2] = 0.0;

  max=ev[maxid];
  mid=ev[midid];
  min=ev[minid];*/
  
  // define "maxis" = molecular axis
  // for prolate take the longest axis
  // for oblate take the shortest axis
  
  double maxis[3];  
  
  if (which == PROLATE) {
    maxis[0]=evectors[0][maxid];
    maxis[1]=evectors[1][maxid];
    maxis[2]=evectors[2][maxid];    
  }
  else if (which == OBLATE) {
    maxis[0]=evectors[0][minid];
    maxis[1]=evectors[1][minid];
    maxis[2]=evectors[2][minid];    
  }   
  
// nematic tensor  
  Q[0]+=3.0*maxis[0]*maxis[0]-1.0;
  Q[1]+=3.0*maxis[1]*maxis[1]-1.0;
  Q[2]+=3.0*maxis[2]*maxis[2]-1.0;
  Q[3]+=3.0*maxis[0]*maxis[1];
  Q[4]+=3.0*maxis[0]*maxis[2];
  Q[5]+=3.0*maxis[1]*maxis[2];
  }
  //divide by Number of molecules * 2
  Q[0]/=2.0*nchunk;
  Q[1]/=2.0*nchunk;
  Q[2]/=2.0*nchunk;
  Q[3]/=2.0*nchunk;
  Q[4]/=2.0*nchunk;
  Q[5]/=2.0*nchunk;

  //diagonalize Q
  Qtensor[0][0]=Q[0];
  Qtensor[1][1]=Q[1];
  Qtensor[2][2]=Q[2];
  Qtensor[0][1]=Q[3];
  Qtensor[1][0]=Q[3];
  Qtensor[0][2]=Q[4];
  Qtensor[2][0]=Q[4];
  Qtensor[1][2]=Q[5];
  Qtensor[2][1]=Q[5];  

  double Qvectors[3][3], Qval[3];
  
  if (MathExtra::jacobi(Qtensor,Qval,Qvectors))
     error->all(FLERR,"Q: Insufficient Jacobi rotations for rigid molecule");   

  // order by largest / smallest eigenvector
  double qmax, qmin, qmid;
  int qmaxid, qminid, qmidid;
  double Qmaxev[3];

  qmaxid=0;
  qminid=0;
  qmidid=0; // CAUTION!! SORTING OF ABSOLUTE VALUES, FOR SIGNED VALUES USE INDICES LATER
  qmax = MAX(fabs(Qval[0]),fabs(Qval[1]));
  qmin = MIN(fabs(Qval[0]),fabs(Qval[1]));
  if (qmax>fabs(Qval[qmaxid])) qmaxid=1;
  if (qmin<fabs(Qval[qminid])) qminid=1;
  qmax = MAX(qmax,fabs(Qval[2]));
  qmin = MIN(qmin,fabs(Qval[2]));
  if (qmax>fabs(Qval[qmaxid])) qmaxid=2;
  if (qmin<fabs(Qval[qminid])) qminid=2;

  if (qmidid==qminid || qmidid==qmaxid) qmidid+=1;
  if (qmidid==qminid || qmidid==qmaxid) qmidid+=1;
  qmid=fabs(Qval[qmidid]);

  //if (Qval[0] < EPSILON*qmax) Qval[0] = 0.0;
  //if (Qval[1] < EPSILON*qmax) Qval[1] = 0.0;
  //if (Qval[2] < EPSILON*qmax) Qval[2] = 0.0;

  //qmax=Qval[qmaxid];
  //qmid=Qval[qmidid];
  //qmin=Qval[qminid];  
  
  //Qmaxev[0]=Qvectors[0][qmaxid];
  //Qmaxev[1]=Qvectors[1][qmaxid];
  //Qmaxev[2]=Qvectors[2][qmaxid]; 
  
  //vector[0]= Qmaxev[0];
  //vector[1]= Qmaxev[1];
  //vector[2]= Qmaxev[2];
  
//  vector[0] = Qvectors[0][qmaxid];
//  vector[1] = Qvectors[1][qmaxid];
//  vector[2] = Qvectors[2][qmaxid];  
//
  vector[0] = Qval[qmaxid];
  vector[1] = Qval[qmidid];
  vector[2] = Qval[qminid];
  vector[3] = Qvectors[0][qmaxid];
  vector[4] = Qvectors[1][qmaxid];
  vector[5] = Qvectors[2][qmaxid];
  vector[6] = Qvectors[0][qmidid];
  vector[7] = Qvectors[1][qmidid];
  vector[8] = Qvectors[2][qmidid];
  vector[9] = Qvectors[0][qminid];
  vector[10] = Qvectors[1][qminid];
  vector[11] = Qvectors[2][qminid];
}


/* ----------------------------------------------------------------------
   calculate per-chunk COM, used by both scalar and tensor
------------------------------------------------------------------------- */

void ComputeNematicChunk::com_chunk()
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
  //if (tensor) size_array_rows = nchunk;
  //else size_vector = nchunk;
  //size_vector = nchunk;
  
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

void ComputeNematicChunk::lock_enable()
{
  cchunk->lockcount++;
}

/* ----------------------------------------------------------------------
   decrement lock counter in compute chunk/atom, it if still exists
------------------------------------------------------------------------- */

void ComputeNematicChunk::lock_disable()
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

int ComputeNematicChunk::lock_length()
{
  nchunk = cchunk->setup_chunks();
  return nchunk;
}

/* ----------------------------------------------------------------------
   set the lock from startstep to stopstep
------------------------------------------------------------------------- */

void ComputeNematicChunk::lock(Fix *fixptr, bigint startstep, bigint stopstep)
{
  cchunk->lock(fixptr,startstep,stopstep);
}

/* ----------------------------------------------------------------------
   unset the lock
------------------------------------------------------------------------- */

void ComputeNematicChunk::unlock(Fix *fixptr)
{
  cchunk->unlock(fixptr);
}

/* ----------------------------------------------------------------------
   free and reallocate per-chunk arrays
------------------------------------------------------------------------- */

void ComputeNematicChunk::allocate()
{
  memory->destroy(massproc);
  memory->destroy(masstotal);
  memory->destroy(com);
  memory->destroy(comall);
  memory->destroy(rg);
  memory->destroy(rgall);
  memory->destroy(rgt);
  memory->destroy(rgtall);
  memory->destroy(Q);   
  maxchunk = nchunk;
  memory->create(massproc,maxchunk,"nematic/chunk:massproc");
  memory->create(masstotal,maxchunk,"nematic/chunk:masstotal");
  memory->create(com,maxchunk,3,"nematic/chunk:com");
  memory->create(comall,maxchunk,3,"nematic/chunk:comall");
  memory->create(rgt,maxchunk,6,"nematic/chunk:rgt");
  memory->create(rgtall,maxchunk,6,"nematic/chunk:rgtall");  
  memory->create(Q,6,"nematic/chunk:Q");  
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */
/*
double ComputeNematicChunk::memory_usage()
{
  double bytes = (bigint) maxchunk * 2 * sizeof(double);
  bytes += (bigint) maxchunk * 2*3 * sizeof(double);
  //if (tensor) bytes += (bigint) maxchunk * 2*9 * sizeof(double);
  //else bytes += (bigint) maxchunk * 2 * sizeof(double);
  bytes += (bigint) maxchunk * 2 * sizeof(double);
  return bytes;
}*/
