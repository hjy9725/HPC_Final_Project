# HPC_Final_Project

Enter the folder named code, 

run `make solver` to get the executable file named "solver" in explicit equation, 

or change the value of `Is_Explicit` from `PETSC_TRUE` to `PETSC_FALSE` 

(in line 73 : `PetscBool	Is_Explicit = PETSC_TRUE` from `solver.cpp`)  

to get the executable file named `solver` in implicit equation.

Options in command line:

Use `-Nodes_Num`  to specify the node size and the default number is 10.

Use `-K_Conductivity` to specify the conductivity and the default value is `1.0`.

Use `-Rho_density` to specify the density and the default value is `1.0`.

Use `-C_Heat_Capacity` to specify the heat capacity and the default value is `1.0`.

Use `-dt` to specify the time_step and the default value is `0.0003125`.

Project details: [HPC_Final_Project_Report.pdf](https://github.com/hjy9725/HPC_Final_Project/blob/main/HPC_Final_Project_Report.pdf)

