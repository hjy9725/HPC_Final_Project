-include ../../../../petscdir.mk
CFLAGS           =
FFLAGS           =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/ksp/ksp/tutorials/
EXAMPLESC        = solver.cpp ex1.cpp explicit.cpp implicit.cpp ex3.cpp restart.c ex_and_implicit_restart.c
MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk
NP               = 1
DIRS             = network amrex

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

