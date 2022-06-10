/* ------------------------------------------------------------------------

   This program solves the one-dimensional heat equation (also called the
   diffusion equation),
       u_t = u_xx,
   on the domain 0 <= x <= 1, with the boundary conditions
       u(t,0) = 0, u(t,1) = 0,
   and the initial condition
       u(0,x) = sin(l*pi*x).
   This is a linear, second-order, parabolic equation.

   We discretize the right-hand side using finite differences with
   uniform grid spacing h:
       u_xx = (u_{i+1} - 2u_{i} + u_{i-1})/(h^2)

   We set a variable named Is_Explicit to indicate whether the method we
   adopt is explicit or implicit.

  ------------------------------------------------------------------------- */
static char help[] = "Solves the one-dimensional heat equation.\n\n";

#include <stdio.h>
#include <assert.h>
#include <petscksp.h>
#include <petscmath.h>
#include <petscsys.h>
#include <petscviewerhdf5.h>
#include <math.h>
#include "hdf5.h"

#define pi acos(-1) 

// solver tools
void Explicit_Euler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, PetscInt Nodes_Num,
	PetscReal dt, PetscReal Time_End, PetscInt Boundary_Condition_x0_h, PetscInt Boundary_Condition_x0_t,
	PetscReal H_F_x0, PetscReal H_F_x1, PetscReal Temp_x0, PetscReal Temp_x1 );

void Implicit_Euler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, KSP ksp, PC pc,
	PetscReal dt, PetscReal Time_End, PetscInt Nodes_Num, PetscInt Boundary_Condition_x0_h, PetscInt Boundary_Condition_x0_t,
	PetscReal H_F_x0, PetscReal H_F_x1, PetscReal Temp_x0, PetscReal Temp_x1 );

void MatBC( MPI_Comm comm, Mat A, PetscInt Nodes_Num, PetscInt Boundary_Condition_x0_h, PetscInt Boundary_Condition_x0_t );

// HDF5 tools
void HDF5_Write();

void HDF5_Read();

int main( int argc, char **argv )
{
	/*---------------------------------------------------

      Declare the variables needed for computation 

      -------------------------------------------------*/
	PetscInitialize(&argc, &argv, (char*)0, help);
	MPI_Comm comm = PETSC_COMM_WORLD;

	// variables
	PetscInt	Nodes_Num 			= 10;
	PetscReal	K_Conductivity 		= 1.0;	
	PetscReal	Rho_density 		= 1.0;	   
	PetscReal	C_Heat_Capacity 	= 1.0;	  
	PetscReal	dt 					= 0.0003125;

	PetscReal	Time_End 			= 1.0;
	PetscReal   Temp_x0				= 0.0;		
	PetscReal   Temp_x1				= 0.0;		
	PetscReal   Heat_Flux_x0		= 1.0;		
	PetscReal   Heat_Flux_x1		= 1.0;		
	PetscInt	Boundary_Condition_x0_h = 0;	// boundary condition at x=0: 0 for prescribed temperature, 1 for heat flux.
	PetscInt	Boundary_Condition_x0_t = 0;	// boundary condition at x=0: 0 for prescribed temperature, 1 for heat flux.
	PetscBool	Is_Explicit = PETSC_TRUE;

	// Varialbes Get From Commond Line
	PetscOptionsGetInt( PETSC_NULL, PETSC_NULL, "-Nodes_Num",		&Nodes_Num , PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-K_Conductivity",	&K_Conductivity, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-Rho_density",		&Rho_density, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-C_Heat_Capacity",	&C_Heat_Capacity, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-dt",				&dt, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-Time_End",		&Time_End, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-Temp_x0",		 	&Temp_x0, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-Temp_x1",		  	&Temp_x1, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-Heat_Flux_x0",	&Heat_Flux_x0, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-Heat_Flux_x1",	&Heat_Flux_x1, PETSC_NULL);

	PetscOptionsGetInt( PETSC_NULL, PETSC_NULL, "-Boundary_Condition_x0_h",			&Boundary_Condition_x0_h, PETSC_NULL);
	PetscOptionsGetInt( PETSC_NULL, PETSC_NULL, "-Boundary_Condition_x0_t",			&Boundary_Condition_x0_t, PETSC_NULL);

	PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-Is_Explicit",		&Is_Explicit, PETSC_NULL);

	PetscReal	dx = 1.0 / Nodes_Num ;

	//Iteration coefficient
	const double Beta = dt / (Rho_density * C_Heat_Capacity);
	const double Alpha = (Beta * K_Conductivity) / (dx * dx);

	const double ll = 1.0;

	// Matrix_A values for x, y
	double * Matrix_A_X = new double[Nodes_Num +1](); 
	for(int ii=1; ii<Nodes_Num +1; ii++) {
		Matrix_A_X[ii] = Matrix_A_X[ii-1] + dx;
	}
	
	// f = sin(l*pi*x)
	double * Heat_Source = new double[Nodes_Num +1]();
	for(int ii=0; ii<Nodes_Num +1; ii++) {
		Heat_Source[ii] = Beta * sin(ll * pi * Matrix_A_X[ii]);
	}

	Vec	ff;
	VecCreate(comm, &ff);
	VecSetSizes(ff, PETSC_DECIDE, Nodes_Num +1);
	VecSetFromOptions(ff);
	for(PetscInt ii=0; ii<Nodes_Num +1; ii++) {
		VecSetValues(ff, 1, &ii, &Heat_Source[ii], INSERT_VALUES);
	}

	VecAssemblyBegin(ff);
	VecAssemblyEnd(ff);

//	VecView(ff, PETSC_VIEWER_STDOUT_(comm));

	// matrix A
	Mat	A;
	MatCreate(comm, &A);
	MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, Nodes_Num +1, Nodes_Num +1);
	MatSetFromOptions(A);
	MatMPIAIJSetPreallocation(A, 3, NULL, 3, NULL);
	MatSeqAIJSetPreallocation(A, 3, NULL);

	PetscInt	rstart, rend, m, n;
	MatGetOwnershipRange(A, &rstart, &rend);
	MatGetSize(A, &m, &n);
	
	for (PetscInt ii=rstart; ii<rend; ii++) 
	{
		PetscInt	index[3] = {ii-1, ii, ii+1};
		PetscScalar	value[3];
		if (Is_Explicit) {
			value[0] = 1.0*Alpha;
			value[1] = 1.0-2.0*Alpha;
			value[2] = 1.0*Alpha;
		}
		else {
			value[0] = -1.0*Alpha;
			value[1] =  2.0*Alpha+1.0;
			value[2] = -1.0*Alpha;
		}

		if (ii == 0) {
			MatSetValues(A, 1, &ii, 2, &index[1], &value[1], INSERT_VALUES);
		}
		else if (ii == n-1) {
			MatSetValues(A, 1, &ii, 2, index, value, INSERT_VALUES);
		}
		else {
			MatSetValues(A, 1, &ii, 3, index, value, INSERT_VALUES);
		}
	}

	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

//	MatView(A, PETSC_VIEWER_STDOUT_WORLD);

	// initial condition: u_0 = exp(x)
	double * u_0 = new double[Nodes_Num +1]();
	for(int ii=0; ii<Nodes_Num +1; ii++) {
		u_0[ii] = exp(Matrix_A_X[ii]);
	}

	Vec	uu;
	VecDuplicate(ff, &uu);
	for(PetscInt ii=0; ii<Nodes_Num +1; ii++) {
		VecSetValues(uu, 1, &ii, &u_0[ii], INSERT_VALUES);
	}

	VecAssemblyBegin(uu);
	VecAssemblyEnd(uu);

    //	VecView(uu, PETSC_VIEWER_STDOUT_(comm));


	// boundary conditions
	MatBC( comm, A, Nodes_Num , Boundary_Condition_x0_h, Boundary_Condition_x0_t );

	// solver
	KSP		ksp;
	PC		pc;
	KSPCreate(comm, &ksp);
	Vec	u_new;
	VecDuplicate(uu, &u_new);

	PetscReal H_F_x0 = (dx * Heat_Flux_x0) / K_Conductivity;
	PetscReal H_F_x1 = (dx * Heat_Flux_x1) / K_Conductivity;
	
	if (Is_Explicit) {
		Explicit_Euler( comm, A, u_new, uu, ff, Nodes_Num , 
			            dt, Time_End, Boundary_Condition_x0_h, Boundary_Condition_x0_t, 
			            H_F_x0, H_F_x1, Temp_x0, Temp_x1 );
	}
	else {
		Implicit_Euler( comm, A, u_new, uu, ff, ksp, pc, dt, Time_End, Nodes_Num , Boundary_Condition_x0_h, Boundary_Condition_x0_t, H_F_x0, H_F_x1, Temp_x0, Heat_Flux_x1 );
	}

	VecView(u_new, PETSC_VIEWER_STDOUT_(comm));

	// destory
	KSPDestroy(&ksp);
	VecDestroy(&uu);
	VecDestroy(&u_new);
	VecDestroy(&ff);
	MatDestroy(&A);

	PetscFinalize();
	delete [] Matrix_A_X; delete [] Heat_Source; delete [] u_0;
	return 0;
}

void Implicit_Euler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, KSP ksp, PC pc,
	PetscReal dt, PetscReal Time_End, PetscInt Nodes_Num, PetscInt Boundary_Condition_x0_h, PetscInt Boundary_Condition_x0_t,
	PetscReal H_F_x0, PetscReal H_F_x1, PetscReal Temp_x0, PetscReal Temp_x1 )
{
	PetscInt	N = Time_End / dt;
	PetscInt	its;
	PetscInt	zero = 0;
	PetscReal	rnorm;
	PetscReal	time = 0.0;

	// Initialize For KSP
	KSPSetOperators(ksp, A, A);
	KSPSetType(ksp, KSPGMRES);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCJACOBI);
	KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPSetFromOptions(ksp);

	// Solving Progress
	//PetscPrintf(comm, "======> Index_For_Time: %D\t time: %g<======\n", (int)zero, (double)time);

	for(int ii=0; ii<N; ii++) {
		time += dt;
		VecAXPY(b, 1.0, f);
		if (Boundary_Condition_x0_h) {
			VecSetValues(b, 1, &zero, &H_F_x0, INSERT_VALUES);
		}
		else {
			VecSetValues(b, 1, &zero, &Temp_x0, INSERT_VALUES);
		}
		if (Boundary_Condition_x0_t) {
			VecSetValues(b, 1, &Nodes_Num , &H_F_x1, INSERT_VALUES);
		}
		else {
			VecSetValues(b, 1, &Nodes_Num , &Temp_x0, INSERT_VALUES);
		}
		VecAssemblyBegin(b);
		VecAssemblyEnd(b);
		KSPSolve(ksp, b, x);
		VecCopy(x, b);
        /*
           VecView(x, PETSC_VIEWER_STDOUT_(comm));
		*/
		KSPMonitor(ksp, its, rnorm);
		//PetscPrintf(comm, "------- Step Solving Process Done. \n");
		//PetscPrintf(comm, "------- Iteration OF KSP :         %D\t r_norm: %g\n", its, (double)rnorm);
		//PetscPrintf(comm, "======> Index_For_Time:            %D\t time: %g<======\n", ii+1, (double)time);
	}
}

void Explicit_Euler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, PetscInt Nodes_Num,
	PetscReal dt, PetscReal Time_End, PetscInt Boundary_Condition_x0_h, PetscInt Boundary_Condition_x0_t,
	PetscReal H_F_x0, PetscReal H_F_x1, PetscReal Temp_x0, PetscReal Temp_x1 )
{
	PetscInt	N = Time_End / dt;
	PetscInt	zero = 0;
	PetscReal	time = 0.0;

	for(int ii=0; ii<N; ii++) {
		time += dt;
		VecAXPY(b, 1.0, f);
		if (Boundary_Condition_x0_h) {
			VecSetValues(b, 1, &zero, &H_F_x0, INSERT_VALUES);
		}
		else {
			VecSetValues(b, 1, &zero, &Temp_x0, INSERT_VALUES);
		}
		if (Boundary_Condition_x0_t) {
			VecSetValues(b, 1, &Nodes_Num , &H_F_x1, INSERT_VALUES);
		}
		else {
			VecSetValues(b, 1, &Nodes_Num , &Temp_x0, INSERT_VALUES);
		}
		VecAssemblyBegin(b);
		VecAssemblyEnd(b);
		MatMult(A, b, x);
		VecCopy(x, b);
		
	}
	PetscPrintf(comm, "======> Explicit_Euler Solving Process Done <======\n");
}

void MatBC( MPI_Comm comm, Mat A, PetscInt Nodes_Num, PetscInt Boundary_Condition_x0_h, PetscInt Boundary_Condition_x0_t )
{
	if (!Boundary_Condition_x0_h) {
		PetscInt	row = 0;
		PetscInt	col[2] = {0, 1};
		PetscScalar	value[2] = {1.0, 0.0};

		MatSetValues(A, 1, &row, 2, col, value, INSERT_VALUES);
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

		//PetscPrintf(comm, "======> Essential BC at x=0: prescribed temperature.\n");
	}
	else {
		PetscInt	row = 0;
		PetscInt	col[2] = {0, 1};
		PetscScalar	value[2] = {1.0, -1.0};

		MatSetValues(A, 1, &row, 2, col, value, INSERT_VALUES);
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

		//PetscPrintf(comm, "======> Natural BC at x=0: heat flux.\n");
	}
	if (!Boundary_Condition_x0_t) {
		PetscInt	row = Nodes_Num ;
		PetscInt	col[2] = {Nodes_Num -1, Nodes_Num };
		PetscScalar	value[2] = {0.0, 1.0};

		MatSetValues(A, 1, &row, 2, col, value, INSERT_VALUES);
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

		//PetscPrintf(comm, "======> Essential BC at x=0: prescribed temperature.\n");
	}
	else {
		PetscInt	row = Nodes_Num ;
		PetscInt	col[2] = {Nodes_Num -1, Nodes_Num };
		PetscScalar	value[2] = {-1.0, 1.0};

		MatSetValues(A, 1, &row, 2, col, value, INSERT_VALUES);
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

		//PetscPrintf(comm, "======> Natural BC at x=1: heat flux.\n");
	}
//	MatView(A, PETSC_VIEWER_STDOUT_WORLD);
}

void HDF5_Write_Solution(Vec uu, PetscInt Nodes_Num, const double Alpha, const double Beta, const double ll,
		PetscInt Boundary_Condition_x0_h, PetscInt Boundary_Condition_x0_t, PetscReal H_F_x0, PetscReal H_F_x1, PetscReal Temp_x0, PetscReal Temp_x1,
		PetscReal dt, PetscReal dx )
{
	// write current solution

    /* Assert that the alpha <= 0.5 */
    
    //PetscReal	alpha = kappa * dt / ( rho * c * delta_x * delta_x );

    //assert ( alpha <= 0.5 );
    
	hid_t	file_id;
	hid_t	dataset_uu, dataset_Alpha, dataset_Beta, dataset_M, dataset_delx, dataset_delt, dataset_ll;
	hid_t	dataset_headbc, dataset_tailbc, dataset_hbch, dataset_hbct, dataset_ggh, dataset_ggt;
	hid_t	dataspace_uu, dataspace_scalar;

	hsize_t	dim_uu[1]; dim_uu[0] = Nodes_Num ;
	hsize_t	dim_scalar[1]; dim_scalar[0] = 1;

	PetscScalar	*array;
	VecGetArray(uu, &array);

	char file[10] = "SOL_";
	char num[10] = "0000";
	strcat(file, num);
	file_id = H5Fcreate(file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	dataspace_uu = H5Screate_simple(1, dim_uu, NULL);
	dataspace_scalar = H5Screate_simple(1, dim_scalar, NULL);

	dataset_uu	= H5Dcreate2(file_id, "/Current_solution", H5T_NATIVE_DOUBLE, dataspace_uu, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_Alpha	= H5Dcreate2(file_id, "/Alpha", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_Beta	= H5Dcreate2(file_id, "/Beta", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_M	= H5Dcreate2(file_id, "/Matrix_size", H5T_NATIVE_INT, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_delx	= H5Dcreate2(file_id, "/dx", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_delt	= H5Dcreate2(file_id, "/dt", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_ll	= H5Dcreate2(file_id, "/Heat_src_para", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_headbc	= H5Dcreate2(file_id, "/Boundary_Condition_x0_h", H5T_NATIVE_INT, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_tailbc	= H5Dcreate2(file_id, "/Boundary_Condition_x0_t", H5T_NATIVE_INT, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_hbch	= H5Dcreate2(file_id, "/Head_hbc", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_hbct	= H5Dcreate2(file_id, "/Tail_hbc", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_ggh	= H5Dcreate2(file_id, "/Head_gbc", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_ggt	= H5Dcreate2(file_id, "/Tail_gbc", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	

	H5Dwrite(dataset_uu, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
	H5Dwrite(dataset_Alpha, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Alpha);
	H5Dwrite(dataset_Beta, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Beta);
	H5Dwrite(dataset_M, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Nodes_Num );
	H5Dwrite(dataset_delx, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dx);
	H5Dwrite(dataset_delt, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dt);
	H5Dwrite(dataset_ll, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &ll);
	H5Dwrite(dataset_headbc, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Boundary_Condition_x0_h);
	H5Dwrite(dataset_tailbc, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Boundary_Condition_x0_t);
	H5Dwrite(dataset_hbch, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &H_F_x0);
	H5Dwrite(dataset_hbct, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &H_F_x1);
	H5Dwrite(dataset_ggh, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Temp_x0);
	H5Dwrite(dataset_ggt, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Temp_x1);

	H5Dclose(dataset_uu);
	H5Dclose(dataset_Alpha);
	H5Dclose(dataset_Beta);
	H5Dclose(dataset_M);
	H5Dclose(dataset_delx);
	H5Dclose(dataset_delt);
	H5Dclose(dataset_ll);
	H5Dclose(dataset_headbc);
	H5Dclose(dataset_tailbc);
	H5Dclose(dataset_hbch);
	H5Dclose(dataset_hbct);
	H5Dclose(dataset_ggh);
	H5Dclose(dataset_ggt);
	H5Sclose(dataspace_uu);
	H5Sclose(dataspace_scalar);

	H5Fclose(file_id);
}

void HDF5_Read();
