#include <omp.h>
#include <stdio.h>

int main(){
    // Begining of parallel region
    # pragma omp parallel
    {
        printf("Hello World from Thread %d!\n",omp_get_thread_num());
    }
    // Ending of parallel region
}
