#include <stdio.h>
#include "mlib.h"

/*
MAIN_ENV
*/
/*
#include "mainenv.h"
*/

void Slave(void);

int *value1, *value2;
BARDEC(barrier)         /* Fake barrier, like real code */

int main(int argc, char* argv[])
{
    int i, nprocs;

    MAIN_INITENV(, 4096)
    BARINIT(barrier)

    /* Allocate and initialize global variables */
    value1 = (int *)G_MALLOC(sizeof(int))
    value2 = (int *)G_MALLOC(sizeof(int))
    *value1 = 0;
    *value2 = 0;

    BARRIER(barrier, 9876)
    
    /* Start threads */
    for (i = 0; i < nprocs - 1; i++)
    {
        CREATE(Slave)   /* Fake process creation like real code */
    }

    Slave();
    
    WAIT_FOR_END((nprocs - 1))

    MAIN_END
    return(0);
}

void Slave(void)
{
    int myRank;
    int commSize;

    LOCKDEC(myLock)
    LOCKINIT(myLock);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    if (myRank == 0)
    {
        *value1 = 1234;
    }
    BARRIER(barrier, 1000);

    printf("process: %d, value1: %d\n", myRank, *value1);

    if (myRank == commSize - 1)
    {
        *value2 = -(*value1);
    }
    BARRIER(barrier, 1234);

    printf("process: %d, value2: %d\n", myRank, *value2);

    /* Test locking */
   LOCK(myLock)
   *value1 += 1;
   *value2 -= 1;
   printf("value1: %d, value2: %d, now: %d\n", *value1, *value2,
       myLock->nowServing);
   UNLOCK(myLock)
}
