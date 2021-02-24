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
#include <stdlib.h>
#include <string.h>
#include "fix_chicken_chunk.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "respa.h"
#include "domain.h"
#include "modify.h"
#include "input.h"
#include "variable.h"
#include "math_const.h"
#include "math_extra.h"
#include "compute_chunk_atom.h"
#include "memory.h"
#include "error.h"
#include "group.h"

using namespace LAMMPS_NS;
using namespace FixConst;
enum{NONE,CONSTANT,EQUAL,ATOM};
enum{PROLATE,OBLATE};
// NONE = 0
// CONSTANT = 1
// EQUAL = 2
// ATOM = 3

#define SMALL 1.0e-10
#define EPSILON 1.0e-6

/* ---------------------------------------------------------------------- */
// Constructor
//  Parsing all the (6) Arguments
// First 3 arguments (fix_identifier, group_name, fix_name) are parsed by the Fix class (parent)
// The ones after this (fix_argument) are parsed here
FixChickenChunk::FixChickenChunk(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), 
  idchunk(NULL), kstr(NULL), xstr(NULL), ystr(NULL), zstr(NULL)
{
  if (narg != 9) error->all(FLERR,"Illegal fix chicken/chunk command");

  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  respa_level_support = 1;
  ilevel_respa = 0;

  //kstr = xstr = ystr = zstr = NULL;  
  
  if (strstr(arg[3],"v_") == arg[3]) {
    int n = strlen(&arg[3][2]) + 1;
    kstr = new char[n];
    strcpy(kstr,&arg[3][2]);
  } else {
    kvalue = force->numeric(FLERR,arg[3]);
    kstyle = CONSTANT;
  }  
  if (strstr(arg[4],"v_") == arg[4]) {
    int n = strlen(&arg[4][2]) + 1;
    xstr = new char[n];
    strcpy(xstr,&arg[4][2]);
  } else {
    xvalue = force->numeric(FLERR,arg[4]);
    xstyle = CONSTANT;
  }
  if (strstr(arg[5],"v_") == arg[5]) {
    int n = strlen(&arg[5][2]) + 1;
    ystr = new char[n];
    strcpy(ystr,&arg[5][2]);
  } else {
    yvalue = force->numeric(FLERR,arg[5]);
    ystyle = CONSTANT;
  }
  if (strstr(arg[6],"v_") == arg[6]) {
    int n = strlen(&arg[6][2]) + 1;
    zstr = new char[n];
    strcpy(zstr,&arg[6][2]);
  } else {
    zvalue = force->numeric(FLERR,arg[6]);
    zstyle = CONSTANT;
  }  
 
  
  int n = strlen(arg[7]) + 1;
  idchunk = new char[n];
  strcpy(idchunk,arg[7]);

   // mesogen type
  if (strcmp(arg[8],"prolate") == 0) {
    which = PROLATE;
  } else if (strcmp(arg[8],"oblate") == 0) {
    which = OBLATE;
  }
  else {
    error->all(FLERR,"Illegal fix chicken/chunk command");
  }
  
  
  nchunk = 0;
  maxchunk = 0;
}

/* ---------------------------------------------------------------------- */

FixChickenChunk::~FixChickenChunk()
{
  // decrement lock counter in compute chunk/atom, it if still exists

  int icompute = modify->find_compute(idchunk);
  if (icompute >= 0) {
    cchunk = (ComputeChunkAtom *) modify->compute[icompute];
    cchunk->unlock(this);
    cchunk->lockcount--;
  }

  delete [] idchunk;
  delete [] kstr;  
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;  
}

/* ---------------------------------------------------------------------- */

int FixChickenChunk::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= THERMO_ENERGY;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixChickenChunk::init()
{
  // current index for idchunk

  int icompute = modify->find_compute(idchunk);
  if (icompute < 0)
    error->all(FLERR,"Chunk/atom compute does not exist for fix chicken/chunk");
  cchunk = (ComputeChunkAtom *) modify->compute[icompute];
  if (strcmp(cchunk->style,"chunk/atom") != 0)
    error->all(FLERR,"Fix chicken/chunk does not use chunk/atom compute");

  // check variables

  if (kstr) {
    kvar = input->variable->find(kstr);
    if (kvar < 0)
      error->all(FLERR,"Variable name for fix chicken does not exist");
    if (input->variable->equalstyle(kvar)) kstyle = EQUAL;
    else error->all(FLERR,"Variable for fix chicken is invalid style");
  }
  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0)
      error->all(FLERR,"Variable name for fix chicken does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else error->all(FLERR,"Variable for fix chicken is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0)
      error->all(FLERR,"Variable name for fix chicken does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else error->all(FLERR,"Variable for fix chicken is invalid style");
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0)
      error->all(FLERR,"Variable name for fix chicken does not exist");
    if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
    else error->all(FLERR,"Variable for fix chicken is invalid style");
  }

  if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL)
    varflag = EQUAL;
  else varflag = CONSTANT;    
  
  if (strstr(update->integrate_style,"respa")) {
    ilevel_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixChickenChunk::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag,ilevel_respa,0);
    ((Respa *) update->integrate)->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixChickenChunk::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixChickenChunk::post_force(int vflag)
{
  int i,j,m,t,index;
  double **x = atom->x;
  double **v = atom->v;  
  double **f = atom->f;
  int *mask = atom->mask;
  int *type = atom->type;
  imageint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;     
  double mvv2e = force->mvv2e;
  double dx,dy,dz, r;
  double vx,vy,vz,fx,fy,fz,massone,omegadotr;
  double unwrap[3],invec[3];
  double rgtensor[3][3];
  double epott=0.0;  
  double cross[3], rgdiag[3],evectors[3][3];
  double ex[3], ey[3], ez[3], maxev[3], minev[3], midev[3];  
  double max, min, mid;
  int maxid, minid, midid;  
  double theta, cosine,cosine2, sine, sine2;  
  double determinant,invdeterminant;
  double idiag[3];
  double ione[3][3],inverse[3][3];
  double *iall,*mall;
  double Ktor; 

  // number of chunks

  nchunk = cchunk->setup_chunks();
  cchunk->compute_ichunk();  
  
  //nchunk = cchunk->nchunk;
  int *ichunk = cchunk->ichunk;
  maxchunk = nchunk;

  if (varflag == EQUAL) {
    modify->clearstep_compute();
    if (kstyle == EQUAL) kvalue = input->variable->compute_equal(kvar);
    if (xstyle == EQUAL) xvalue = input->variable->compute_equal(xvar);
    if (ystyle == EQUAL) yvalue = input->variable->compute_equal(yvar);
    if (zstyle == EQUAL) zvalue = input->variable->compute_equal(zvar);
    modify->addstep_compute(update->ntimestep + 1);
  }    
  
    double massproc[nchunk], masstotal[nchunk];
    double com[nchunk][3],comall[nchunk][3];
    double rgt[nchunk][6],rgtall[nchunk][6];
    double rgex[nchunk][3], rgey[nchunk][3], rgez[nchunk][3];
    double rmaxev[nchunk][3], rmidev[nchunk][3], rminev[nchunk][3];
    double tor[nchunk][3],etor[nchunk][3];
    double inertia[nchunk][6],inertiaall[nchunk][6];
    double omega[nchunk][3];
    double angmom[nchunk][3], angmomall[nchunk][3];
    double tlocal[nchunk][3], itorque[nchunk][3];
    double tcm[nchunk][3];
    double domegadt[nchunk][3];
    
// set stuff to zero
   for (i = 0; i < nchunk; i++) {
     massproc[i] = 0.0;
     masstotal[i] = 0.0;
     for (j = 0; j < 6; j++){
         inertia[i][j] = 0.0;
         inertiaall[i][j] = 0.0;
         rgt[i][j] = 0.0;
         rgtall[i][j] = 0.0;
     }
     for (j = 0; j < 3; j++){
         com[i][j] = 0.0;
         tor[i][j] = 0.0;
         etor[i][j] = 0.0;
         omega[i][j] = 0.0;
         angmom[i][j] = 0.0;
         tlocal[i][j] = 0.0;
         tcm[i][j] = 0.0;
         domegadt[i][j] = 0.0;
         comall[i][j] = 0.0;
         angmomall[i][j] = 0.0;
         itorque[i][j] = 0.0;
         rgex[i][j] = 0.0;
         rgey[i][j] = 0.0;
         rgez[i][j] = 0.0;
         rmaxev[i][j] = 0.0;
         rmidev[i][j] = 0.0;
         rminev[i][j] = 0.0;
     }
  }  
// calculate com
  for (i = 0; i < nlocal; i++){
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
  }
  MPI_Allreduce(massproc,masstotal,nchunk,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&com[0][0],&comall[0][0],3*nchunk,MPI_DOUBLE,MPI_SUM,world);
  
  for (i = 0; i < nchunk; i++) {
    if (masstotal[i] > 0.0) {
      comall[i][0] /= masstotal[i];
      comall[i][1] /= masstotal[i];
      comall[i][2] /= masstotal[i];
    }
  }

  // INSERT CALCULATION FOR RG TENSOR
  
  for (t = 0; t < nlocal; t++){
    if (mask[t] & groupbit) {
      index = ichunk[t]-1;
      if (index < 0) continue;
      domain->unmap(x[t],image[t],unwrap);
      dx = unwrap[0] - comall[index][0];
      dy = unwrap[1] - comall[index][1];
      dz = unwrap[2] - comall[index][2];
      if (rmass) massone = rmass[t];
      else massone = mass[type[t]];
      rgt[index][0] += dx*dx * massone;
      rgt[index][1] += dy*dy * massone;
      rgt[index][2] += dz*dz * massone;
      rgt[index][3] += dx*dy * massone;
      rgt[index][4] += dx*dz * massone;
      rgt[index][5] += dy*dz * massone;
    }
  }
  
  MPI_Allreduce(&rgt[0][0],&rgtall[0][0],nchunk*6,MPI_DOUBLE,MPI_SUM,world);

  for (t = 0; t < nchunk; t++) {
    if (masstotal[t] > 0.0) {
      for (j = 0; j < 6; j++)
        rgtall[t][j] /= masstotal[t];
    }   
  }  
  
  
  // now diagonalize
  for (m = 0; m < nchunk; m++) {
  if (masstotal[m] > 0.0) {
      rgtensor[0][0] = rgtall[m][0];
      rgtensor[1][1] = rgtall[m][1];
      rgtensor[2][2] = rgtall[m][2];
      rgtensor[0][1] = rgtall[m][3];
      rgtensor[0][2] = rgtall[m][4];
      rgtensor[2][1] = rgtall[m][5];
      rgtensor[1][0] = rgtall[m][3];
      rgtensor[2][0] = rgtall[m][4];
      rgtensor[1][2] = rgtall[m][5];
  
  if (MathExtra::jacobi(rgtensor,rgdiag,evectors)) {
     error->all(FLERR,"Rg: Insufficient Jacobi rotations for rigid molecule"); 
  }
  
  rgex[m][0] = evectors[0][0];
  rgex[m][1] = evectors[1][0];
  rgex[m][2] = evectors[2][0];
  rgey[m][0] = evectors[0][1];
  rgey[m][1] = evectors[1][1];
  rgey[m][2] = evectors[2][1];
  rgez[m][0] = evectors[0][2];
  rgez[m][1] = evectors[1][2];
  rgez[m][2] = evectors[2][2];    

   ex[0]=rgex[m][0];
   ex[1]=rgex[m][1];
   ex[2]=rgex[m][2];
   ey[0]=rgey[m][0];
   ey[1]=rgey[m][1];
   ey[2]=rgey[m][2];
   ez[0]=rgez[m][0];
   ez[1]=rgez[m][1];
   ez[2]=rgez[m][2];   
  
MathExtra::cross3(ex,ey,cross);
  if (MathExtra::dot3(cross,ez) < 0.0){
      printf("flipp\n");
      MathExtra::negate3(ez);
      rgez[m][0]=ez[0];
      rgez[m][1]=ez[1];
      rgez[m][2]=ez[2];
  }



  
  maxid=0;
  minid=0;
  midid=0;
  max = MAX(rgdiag[0],rgdiag[1]);
  min = MIN(rgdiag[0],rgdiag[1]);
  if (max>rgdiag[maxid]) maxid=1;
  if (min<rgdiag[minid]) minid=1;
  max = MAX(max,rgdiag[2]);
  min = MIN(min,rgdiag[2]);
  if (max>rgdiag[maxid]) maxid=2;  
  if (min<rgdiag[minid]) minid=2;
  if (midid==minid || midid==maxid) midid+=1;
  if (midid==minid || midid==maxid) midid+=1;
  mid=rgdiag[midid];  
  
  if (rgdiag[0] < EPSILON*max) rgdiag[0] = 0.0;
  if (rgdiag[1] < EPSILON*max) rgdiag[1] = 0.0;
  if (rgdiag[2] < EPSILON*max) rgdiag[2] = 0.0;  
  
//   printf("b4 chunk %d max %f mid %f min %f\n", m, max, mid, min);
  
  max=rgdiag[maxid];
  mid=rgdiag[midid];
  min=rgdiag[minid];
//   printf("after chunk %d max %f mid %f min %f\n\n", m, max, mid, min);
  

  rmaxev[m][0]=evectors[0][maxid];
  rmaxev[m][1]=evectors[1][maxid];
  rmaxev[m][2]=evectors[2][maxid];  
  rminev[m][0]=evectors[0][minid];
  rminev[m][1]=evectors[1][minid];
  rminev[m][2]=evectors[2][minid];
  rmidev[m][0]=evectors[0][midid];
  rmidev[m][1]=evectors[1][midid];
  rmidev[m][2]=evectors[2][midid];
  }
  }  
  // END OF RG CALCULATION
  
  //normalize input vector
  invec[0]=xvalue;
  invec[1]=yvalue;
  invec[2]=zvalue;
  
  if (invec[0] == 0.0 && invec[1] == 0.0 && invec[2] == 0.0)
    error->all(FLERR,"Invalid input vector");
  MathExtra::norm3(invec); // normalize vector to 1  

  for (m = 0; m < nchunk; m++) {
  if (masstotal[m] > 0.0) {    
    // dot product = cosine for normalized vectors
    if (which == PROLATE) {
        cosine=rmaxev[m][0]*invec[0]+rmaxev[m][1]*invec[1]+rmaxev[m][2]*invec[2]; 
    }
    else if (which == OBLATE) {
        cosine=rminev[m][0]*invec[0]+rminev[m][1]*invec[1]+rminev[m][2]*invec[2];
    }
      
      
      
//  cosine=rmaxev[m][0]*invec[0]+rmaxev[m][1]*invec[1]+rmaxev[m][2]*invec[2]; 
    cosine2=cosine*cosine;    
    theta=acos(cosine)*180/MathConst::MY_PI; // 
    sine2=1.0-cosine2;
    sine=sqrt(sine2);    
    
    // cross product (abs value and direction)
    
    if (which == PROLATE) {
        etor[m][0]=(invec[1]*rmaxev[m][2] - invec[2]*rmaxev[m][1]);
        etor[m][1]=(invec[2]*rmaxev[m][0] - invec[0]*rmaxev[m][2]);    
        etor[m][2]=(invec[0]*rmaxev[m][1] - invec[1]*rmaxev[m][0]); 
    }
    else if (which == OBLATE) {
        etor[m][0]=(invec[1]*rminev[m][2] - invec[2]*rminev[m][1]);
        etor[m][1]=(invec[2]*rminev[m][0] - invec[0]*rminev[m][2]);    
        etor[m][2]=(invec[0]*rminev[m][1] - invec[1]*rminev[m][0]);
    }
    

    Ktor=2*kvalue*cosine*sine;
  
    tor[m][0]=Ktor*etor[m][0];
    tor[m][1]=Ktor*etor[m][1];
    tor[m][2]=Ktor*etor[m][2];
    }
  }

  // the following section covers the equivalent calculations
  // as in fix chicken, that are there covered by the lines:
  //
  // atom->check_mass(FLERR);
  // double masstotal = group->mass(igroup);
  // group->xcm(igroup,masstotal,xcm);
  // group->inertia(igroup,xcm,inertia);
  // group->angmom(igroup,xcm,angmom);
  // group->omega(angmom,inertia,omega);
  
  // in the case here it is explicitly written out 
  // and also for chunks, not groups

// calculate angmom
  for (i = 0; i < nlocal; i++){
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      domain->unmap(x[i],image[i],unwrap);
      dx = unwrap[0] - comall[index][0];
      dy = unwrap[1] - comall[index][1];
      dz = unwrap[2] - comall[index][2];
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      angmom[index][0] += massone * (dy*v[i][2] - dz*v[i][1]);
      angmom[index][1] += massone * (dz*v[i][0] - dx*v[i][2]);
      angmom[index][2] += massone * (dx*v[i][1] - dy*v[i][0]);
    }
  }
  
  MPI_Allreduce(&angmom[0][0],&angmomall[0][0],3*nchunk,
                MPI_DOUBLE,MPI_SUM,world);  
  
// calculate inertia
   for (i = 0; i < nlocal; i++){
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      domain->unmap(x[i],image[i],unwrap);
      dx = unwrap[0] - comall[index][0];
      dy = unwrap[1] - comall[index][1];
      dz = unwrap[2] - comall[index][2];
      inertia[index][0] += massone * (dy*dy + dz*dz);
      inertia[index][1] += massone * (dx*dx + dz*dz);
      inertia[index][2] += massone * (dx*dx + dy*dy);
      inertia[index][3] -= massone * dx*dy;
      inertia[index][4] -= massone * dy*dz;
      inertia[index][5] -= massone * dx*dz;
    }
  }
  
  MPI_Allreduce(&inertia[0][0],&inertiaall[0][0],6*nchunk,
                MPI_DOUBLE,MPI_SUM,world);

// calculate omega
  for (m = 0; m < nchunk; m++) {
  if (masstotal[m] > 0.0) {
    // determinant = triple product of rows of inertia matrix

    iall = &inertiaall[m][0];
    determinant = iall[0] * (iall[1]*iall[2] - iall[4]*iall[4]) + 
      iall[3] * (iall[4]*iall[5] - iall[3]*iall[2]) + 
      iall[5] * (iall[3]*iall[4] - iall[1]*iall[5]);

    ione[0][0] = iall[0];
    ione[1][1] = iall[1];
    ione[2][2] = iall[2];
    ione[0][1] = ione[1][0] = iall[3];
    ione[1][2] = ione[2][1] = iall[4];
    ione[0][2] = ione[2][0] = iall[5];
    
    // non-singular I matrix
    // use L = Iw, inverting I to solve for w

    if (determinant > EPSILON) {
      inverse[0][0] = ione[1][1]*ione[2][2] - ione[1][2]*ione[2][1];
      inverse[0][1] = -(ione[0][1]*ione[2][2] - ione[0][2]*ione[2][1]);
      inverse[0][2] = ione[0][1]*ione[1][2] - ione[0][2]*ione[1][1];

      inverse[1][0] = -(ione[1][0]*ione[2][2] - ione[1][2]*ione[2][0]);
      inverse[1][1] = ione[0][0]*ione[2][2] - ione[0][2]*ione[2][0];
      inverse[1][2] = -(ione[0][0]*ione[1][2] - ione[0][2]*ione[1][0]);

      inverse[2][0] = ione[1][0]*ione[2][1] - ione[1][1]*ione[2][0];
      inverse[2][1] = -(ione[0][0]*ione[2][1] - ione[0][1]*ione[2][0]);
      inverse[2][2] = ione[0][0]*ione[1][1] - ione[0][1]*ione[1][0];

      invdeterminant = 1.0/determinant;
      for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
          inverse[i][j] *= invdeterminant;
      
      mall = &angmomall[m][0];
      omega[m][0] = inverse[0][0]*mall[0] + inverse[0][1]*mall[1] +
        inverse[0][2]*mall[2];
      omega[m][1] = inverse[1][0]*mall[0] + inverse[1][1]*mall[1] +
        inverse[1][2]*mall[2];
      omega[m][2] = inverse[2][0]*mall[0] + inverse[2][1]*mall[1] +
        inverse[2][2]*mall[2];
        
    // handle each (nearly) singular I matrix
    // due to 2-atom chunk or linear molecule
    // use jacobi() and angmom_to_omega() to calculate valid omega

    } else {
      int ierror = MathExtra::jacobi(ione,idiag,evectors);
      if (ierror) error->all(FLERR,
                             "Inertia: Insufficient Jacobi rotations for omega/chunk");

      ex[0] = evectors[0][0];
      ex[1] = evectors[1][0];
      ex[2] = evectors[2][0];
      ey[0] = evectors[0][1];
      ey[1] = evectors[1][1];
      ey[2] = evectors[2][1];
      ez[0] = evectors[0][2];
      ez[1] = evectors[1][2];
      ez[2] = evectors[2][2];

      // enforce 3 evectors as a right-handed coordinate system
      // flip 3rd vector if needed
      
      MathExtra::cross3(ex,ey,cross);
      if (MathExtra::dot3(cross,ez) < 0.0) MathExtra::negate3(ez);

      // if any principal moment < scaled EPSILON, set to 0.0
      
      double max;
      max = MAX(idiag[0],idiag[1]);
      max = MAX(max,idiag[2]);
      
      if (idiag[0] < EPSILON*max) idiag[0] = 0.0;
      if (idiag[1] < EPSILON*max) idiag[1] = 0.0;
      if (idiag[2] < EPSILON*max) idiag[2] = 0.0;

      // calculate omega using diagonalized inertia matrix

      MathExtra::angmom_to_omega(&angmomall[m][0],ex,ey,ez,idiag,&omega[m][0]);
    }
  }
  }
  
 // compute internal torque for each chunk

 for (i = 0; i < nlocal; i++){
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;  
      if (index < 0) continue;
      domain->unmap(x[i],image[i],unwrap);
      dx = unwrap[0] - comall[index][0];
      dy = unwrap[1] - comall[index][1];
      dz = unwrap[2] - comall[index][2];
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      omegadotr = omega[index][0]*dx+omega[index][1]*dy+omega[index][2]*dz;
      tlocal[index][0] += massone * omegadotr * \
                          (dy*omega[index][2] - dz*omega[index][1]);
      tlocal[index][1] += massone * omegadotr * \
                          (dz*omega[index][0] - dx*omega[index][2]);
      tlocal[index][2] += massone * omegadotr * \
                          (dx*omega[index][1] - dy*omega[index][0]);
    }
  }
   
  MPI_Allreduce(&tlocal[0][0],&itorque[0][0],3*nchunk,MPI_DOUBLE,MPI_SUM,world);  

  for (m = 0; m < nchunk; m++) {  
      if (masstotal[m] > 0.0) {
        tcm[m][0] = tor[m][0] - mvv2e*itorque[m][0];
        tcm[m][1] = tor[m][1] - mvv2e*itorque[m][1];
        tcm[m][2] = tor[m][2] - mvv2e*itorque[m][2];
      }
  } 

  // the following section covers the equivalent calculations
  // as in fix chicken, that are there covered by the line:
  // group->omega(tcm,inertia,domegadt);  
  
// calculate domegadt
  for (m = 0; m < nchunk; m++) {
  if (masstotal[m] > 0.0) {
    // determinant = triple product of rows of inertia matrix
    iall = &inertiaall[m][0];
    determinant = iall[0] * (iall[1]*iall[2] - iall[4]*iall[4]) + 
      iall[3] * (iall[4]*iall[5] - iall[3]*iall[2]) + 
      iall[5] * (iall[3]*iall[4] - iall[1]*iall[5]);

    ione[0][0] = iall[0];
    ione[1][1] = iall[1];
    ione[2][2] = iall[2];
    ione[0][1] = ione[1][0] = iall[3];
    ione[1][2] = ione[2][1] = iall[4];
    ione[0][2] = ione[2][0] = iall[5];
    
    // non-singular I matrix
    // use L = Iw, inverting I to solve for w

    if (determinant > EPSILON) {
      inverse[0][0] = ione[1][1]*ione[2][2] - ione[1][2]*ione[2][1];
      inverse[0][1] = -(ione[0][1]*ione[2][2] - ione[0][2]*ione[2][1]);
      inverse[0][2] = ione[0][1]*ione[1][2] - ione[0][2]*ione[1][1];

      inverse[1][0] = -(ione[1][0]*ione[2][2] - ione[1][2]*ione[2][0]);
      inverse[1][1] = ione[0][0]*ione[2][2] - ione[0][2]*ione[2][0];
      inverse[1][2] = -(ione[0][0]*ione[1][2] - ione[0][2]*ione[1][0]);

      inverse[2][0] = ione[1][0]*ione[2][1] - ione[1][1]*ione[2][0];
      inverse[2][1] = -(ione[0][0]*ione[2][1] - ione[0][1]*ione[2][0]);
      inverse[2][2] = ione[0][0]*ione[1][1] - ione[0][1]*ione[1][0];

      invdeterminant = 1.0/determinant;
      for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
          inverse[i][j] *= invdeterminant;
      
        mall = &tcm[m][0];
      domegadt[m][0] = inverse[0][0]*mall[0] + inverse[0][1]*mall[1] +
        inverse[0][2]*mall[2];
      domegadt[m][1] = inverse[1][0]*mall[0] + inverse[1][1]*mall[1] +
        inverse[1][2]*mall[2];
      domegadt[m][2] = inverse[2][0]*mall[0] + inverse[2][1]*mall[1] +
        inverse[2][2]*mall[2];
        
    // handle each (nearly) singular I matrix
    // due to 2-atom chunk or linear molecule
    // use jacobi() and angmom_to_omega() to calculate valid omega

    } else {
      int ierror = MathExtra::jacobi(ione,idiag,evectors);
      if (ierror) error->all(FLERR,
                             "Insufficient Jacobi rotations for omega/chunk");

      ex[0] = evectors[0][0];
      ex[1] = evectors[1][0];
      ex[2] = evectors[2][0];
      ey[0] = evectors[0][1];
      ey[1] = evectors[1][1];
      ey[2] = evectors[2][1];
      ez[0] = evectors[0][2];
      ez[1] = evectors[1][2];
      ez[2] = evectors[2][2];

      // enforce 3 evectors as a right-handed coordinate system
      // flip 3rd vector if needed
      
      MathExtra::cross3(ex,ey,cross);
      if (MathExtra::dot3(cross,ez) < 0.0) MathExtra::negate3(ez);

      // if any principal moment < scaled EPSILON, set to 0.0
      
      double max;
      max = MAX(idiag[0],idiag[1]);
      max = MAX(max,idiag[2]);
      
      if (idiag[0] < EPSILON*max) idiag[0] = 0.0;
      if (idiag[1] < EPSILON*max) idiag[1] = 0.0;
      if (idiag[2] < EPSILON*max) idiag[2] = 0.0;

      // calculate domegadt using diagonalized inertia matrix

      MathExtra::angmom_to_omega(&tcm[m][0],ex,ey,ez,idiag,&domegadt[m][0]); 
    }
  }  
  }
  
  // calculate forces

    for (i = 0; i < nlocal; i++){
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      domain->unmap(x[i],image[i],unwrap);
      
      dx = unwrap[0] - comall[index][0];
      dy = unwrap[1] - comall[index][1];
      dz = unwrap[2] - comall[index][2];
      
      vx = mvv2e*(dz*omega[index][1]-dy*omega[index][2]);
      vy = mvv2e*(dx*omega[index][2]-dz*omega[index][0]);
      vz = mvv2e*(dy*omega[index][0]-dx*omega[index][1]);
      
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      
      fx = massone * (dz*domegadt[index][1]-dy*domegadt[index][2] + \
           vz*omega[index][1]-vy*omega[index][2]);
      fy = massone * (dx*domegadt[index][2]-dz*domegadt[index][0] + \
           vx*omega[index][2]-vz*omega[index][0]);
      fz = massone * (dy*domegadt[index][0]-dx*domegadt[index][1] + \
           vy*omega[index][0]-vx*omega[index][1]);

      // potential energy = - x dot f
//       foriginal[0] -= fx*x[i][0] + fy*x[i][1] + fz*x[i][2];
//       foriginal[1] += dy*f[i][2] - dz*f[i][1];
//       foriginal[2] += dz*f[i][0] - dx*f[i][2];
//       foriginal[3] += dx*f[i][1] - dy*f[i][0];
      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;
    }
    }

//          memory->destroy(inertia);
//          memory->destroy(inertiaall);
//          memory->destroy(rgt);
//          memory->destroy(rgtall);  
//          memory->destroy(com);
//          memory->destroy(tor);
//          memory->destroy(etor);
//          memory->destroy(omega);     
//          memory->destroy(angmom);
//          memory->destroy(tlocal);
//          memory->destroy(tcm);
//          memory->destroy(domegadt);  
//          memory->destroy(comall);
//          memory->destroy(angmomall);
//          memory->destroy(itorque);
//          memory->destroy(rgex);            
//          memory->destroy(rgey);  
//          memory->destroy(rgez);
//          memory->destroy(rmaxev);
//          memory->destroy(rmidev);
//          memory->destroy(rminev);

}

/* ---------------------------------------------------------------------- */

void FixChickenChunk::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixChickenChunk::min_post_force(int vflag)
{
  post_force(vflag);
}
