#include <stdio.h>
#include <mpi.h>

main(int argc, char **argv)
{
    int ierr, num_process, my_id;
    ierr=MPI_Init(&argc, &argv);
    ierr=MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    ierr=MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    
    printf("Hello World from rank %i from %i processes!\n", my_id, num_proess);
    ierr=MPI_Finalize();
}
