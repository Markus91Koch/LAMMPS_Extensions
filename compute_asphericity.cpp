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
#include "compute_asphericity.h"
#include "update.h"
#include "atom.h"
#include "group.h"
#include "domain.h"
#include "error.h"
#include "math_extra.h"

using namespace LAMMPS_NS;

#define EPSILON 1.0e-6

/* ---------------------------------------------------------------------- */

ComputeAsphericity::ComputeAsphericity(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute gyration command");

  scalar_flag = vector_flag = 1;
  size_vector = 3;
  extscalar = 0;
  extvector = 0;

  vector = new double[3];
}

/* ---------------------------------------------------------------------- */

ComputeAsphericity::~ComputeAsphericity()
{
  delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeAsphericity::init()
{
  masstotal = group->mass(igroup);
}

/* ---------------------------------------------------------------------- */

double ComputeAsphericity::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  double xcm[3];
  if (group->dynamic[igroup]) masstotal = group->mass(igroup);
  group->xcm(igroup,masstotal,xcm);
  scalar = group->gyration(igroup,masstotal,xcm);
  return scalar;
}

/* ----------------------------------------------------------------------
   compute the radius-of-gyration tensor of group of atoms
   around center-of-mass cm
   must unwrap atoms to compute Rg tensor correctly
   diagonalize the tensor, find out eigenvectors and eigenvalues
   sort eigenvalues by size (max, mid, min) and accordingly with
   eigenvalues, then output all of those
------------------------------------------------------------------------- */

void ComputeAsphericity::compute_vector()
{
  invoked_vector = update->ntimestep;

  double xcm[3];
  group->xcm(igroup,masstotal,xcm);

  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;
  imageint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  double dx,dy,dz,massone;
  double unwrap[3];

  double rg[6];
  rg[0] = rg[1] = rg[2] = rg[3] = rg[4] = rg[5] = 0.0;
  double rgt[6];
  rgt[0] = rgt[1] = rgt[2] = rgt[3] = rgt[4] = rgt[5] = 0.0;
  double rgtensor[3][3];
  double evectors[3][3], ev[3];
//   double rgex[3],rgey[3],rgez[3];
  
  double max, min, mid;
  int maxid, minid, midid;
  
  
  

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];

      domain->unmap(x[i],image[i],unwrap);
      dx = unwrap[0] - xcm[0];
      dy = unwrap[1] - xcm[1];
      dz = unwrap[2] - xcm[2];

      rg[0] += dx*dx * massone;
      rg[1] += dy*dy * massone;
      rg[2] += dz*dz * massone;
      rg[3] += dx*dy * massone;
      rg[4] += dx*dz * massone;
      rg[5] += dy*dz * massone;
    }
  //MPI_Allreduce(rg,vector,6,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(rg,rgt,6,MPI_DOUBLE,MPI_SUM,world);

  if (masstotal > 0.0) 
    for (int i = 0; i < 6; i++)
       rgt[i] /= masstotal;  
      //vector[i] /= masstotal;
    
    rgtensor[0][0] = rgt[0];
    rgtensor[1][1] = rgt[1];
    rgtensor[2][2] = rgt[2];
    rgtensor[0][1] = rgt[3];
    rgtensor[0][2] = rgt[4];
    rgtensor[2][1] = rgt[5];
    rgtensor[1][0] = rgt[3];
    rgtensor[2][0] = rgt[4];
    rgtensor[1][2] = rgt[5];
    
  if (MathExtra::jacobi(rgtensor,ev,evectors))
     error->all(FLERR,"Insufficient Jacobi rotations for rigid molecule");

//   rgex[0] = evectors[0][0];
//   rgex[1] = evectors[1][0];
//   rgex[2] = evectors[2][0];
//   rgey[0] = evectors[0][1];
//   rgey[1] = evectors[1][1];
//   rgey[2] = evectors[2][1];
//   rgez[0] = evectors[0][2];
//   rgez[1] = evectors[1][2];
//   rgez[2] = evectors[2][2];
// 
//   MathExtra::cross3(rgex,rgey,cross);
//   if (MathExtra::dot3(cross,rgez) < 0.0){
//       printf("flipp\n");
//       MathExtra::negate3(rgez);
// //       rgez[m][0]=ez[0];
// //       rgez[m][1]=ez[1];
// //       rgez[m][2]=ez[2];      
//       
//   } 
//   


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
  
  if (ev[0] < EPSILON*max) ev[0] = 0.0;
  if (ev[1] < EPSILON*max) ev[1] = 0.0;
  if (ev[2] < EPSILON*max) ev[2] = 0.0;  

// asphericity parameter b  
  vector[0] = max - 0.5*(mid + min);
  
// relative shape anisotropy  k_squared = A_3
  vector[1] = 1.0 - 3.0*(max*mid + mid*min + max*min)/((max+mid+max)*(max+mid+max));
  
// radius of gyration  
  vector[2] = max + min + mid;
//   vector[0] = ev[maxid];
//   vector[1] = ev[midid];
//   vector[2] = ev[minid];
//   
//   vector[3] = evectors[0][maxid];
//   vector[4] = evectors[1][maxid];
//   vector[5] = evectors[2][maxid];
//   
//   vector[6] = evectors[0][midid];
//   vector[7] = evectors[1][midid];
//   vector[8] = evectors[2][midid];
//   
//   vector[9] = evectors[0][minid];
//   vector[10] = evectors[1][minid];
//   vector[11] = evectors[2][minid];
  
  /*sorted_evec[0]=evectors[0][maxid];
  sorted_evec[1]=evectors[1][maxid];
  sorted_evec[2]=evectors[2][maxid];
  
  sorted_evec[3]=evectors[0][midid];
  sorted_evec[4]=evectors[1][midid];
  sorted_evec[5]=evectors[2][midid];
  
  sorted_evec[6]=evectors[0][minid];
  sorted_evec[7]=evectors[1][minid];
  sorted_evec[8]=evectors[2][minid];
  
  sorted_evec[9]=max;
  sorted_evec[10]=mid;
  sorted_evec[11]=min;   
  */
  
}
