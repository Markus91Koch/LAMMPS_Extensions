/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(chicken/chunk,FixChickenChunk)

#else

#ifndef LMP_FIX_CHICKEN_CHUNK_H
#define LMP_FIX_CHICKEN_CHUNK_H

#include "fix.h"

namespace LAMMPS_NS {

class FixChickenChunk : public Fix {
 public:
  FixChickenChunk(class LAMMPS *, int, char **);
  ~FixChickenChunk();
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);

 private:
  int ilevel_respa;
  char *idchunk;
  int which;
  //int nchunk;
  int nchunk,maxchunk;
  int varflag;
  int kvar, xvar, yvar, zvar;
  int kstyle,xstyle,ystyle,zstyle;
  char *kstr,*xstr,*ystr,*zstr;
  double kvalue,xvalue,yvalue,zvalue;

  class ComputeChunkAtom *cchunk;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix chicken couple group ID does not exist

Self-explanatory.

E: Two groups cannot be the same in fix chicken couple

Self-explanatory.

*/
