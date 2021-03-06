#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h> 

#define SEED     921
#define NUM_ITER 1000000000

int main(int argc, char* argv[])
{
    //int count = 0;
    int local_count=0, flip=1<<24;
    int rank, num_ranks, i, iter, provided,gap,step; 
    double x, y, z, pi;
    int global_count=0;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    
    double start_time, stop_time, elapsed_time;
    start_time=MPI_Wtime(); //
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks); 
    


    srand(time(NULL)+123456789+rank*100); 
    
    flip=flip/num_ranks;  
    
    // Calculate PI following a Monte Carlo method
    for (int iter = 0; iter < flip; iter++)
    {
        // Generate random (X,Y) points
        x = (double)random() / (double)RAND_MAX;
        y = (double)random() / (double)RAND_MAX;
        z = sqrt((x*x) + (y*y));
        
        // Check if point is in unit circle
        if (z <= 1.0)
        {
            local_count++;
        }
    }

    MPI_Status status[num_ranks];
    gap = 2;
        for (i=0;i<num_ranks;i=i+1)
                {
                if (i%gap == 0 )
                    {
                        int counts;
                        MPI_Recv(&counts, 1, MPI_INT, i, 0, MPI_COMM_WORLD,&status[i]);
                        local_count = local_count + counts;
                        // Estimate Pi and display the result
                    }
                else
                    {
                        MPI_Send(&local_count, 1, MPI_INT, i+1, 0, MPI_COMM_WORLD);
                    }
         }
    for (gap=4;gap<num_ranks+1;gap=gap*2)
        {
            for (step=0;step<num_ranks/2 +1 ;step = step+gap/2)
                {
                    if (step%gap == 0 )
                        {
                            int counts;
                            MPI_Recv(&counts, 1, MPI_INT, step, 0, MPI_COMM_WORLD,&status[i]);
                            local_count = local_count + counts;
                            // Estimate Pi and display the result
                        }
                    else
                        {
                            MPI_Send(&local_count, 1, MPI_INT, step+gap/2, 0, MPI_COMM_WORLD);
                        }
                }
        }
    
        if (rank == 0 )
        {
           global_count = global_count+local_count;
        }
        
        pi = ((double)global_count / (double)(flip*num_ranks)) * 4.0;
        



    stop_time=MPI_Wtime();
    elapsed_time=stop_time-start_time;
    if (rank==0){
        printf("pi: %f\n", pi);
        printf("Execution Time: %f\n", elapsed_time);
    }
    MPI_Finalize();
    return 0;

}

