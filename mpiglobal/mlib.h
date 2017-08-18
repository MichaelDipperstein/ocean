#include <mpi.h>
#include <sys/time.h>

/* Replacements for some of the macros used by parmacs */

/* Won't work with CPP #include must manually edit
#define MAIN_ENV        #include "mainenv.h"
*/
#define MAIN_ENV
#define MAIN_INITENV(X, Y)    MainInit(&argc, &argv, Y, &nprocs);
#define MAIN_END        MainEnd();

/* Won't work with CPP #include must manually edit
#define EXTERN_ENV      #include "extenv.h"
*/
#define EXTERN_ENV

#define CREATE(X)
#define WAIT_FOR_END(X) MPI_Barrier(MPI_COMM_WORLD); /* need to sync */

#define LOCKDEC(X)      LOCK_TYPE *X;
#define LOCKINIT(X)     X = InitializeLock();
#define LOCK(X)         Lock(X);
#define UNLOCK(X)       Unlock(X);

#define BARDEC(X)       char X;     /* Need something so struct will work */
#define BARINIT(X)
#define BARRIER(X, Y)   MPI_Barrier(MPI_COMM_WORLD);

#define G_MALLOC(X)     GlobalMalloc(X);
#define CLOCK(X)        Clock(&X);

typedef struct
{
    int nextAvailable;
    int nowServing;
} LOCK_TYPE;

/****************************************************************************
*                                PROTOTYPES
****************************************************************************/
void MainInit(int *argc, char **argv[], int y, int *nprocs);
void MainEnd(void);

/* Memory stuff */

/* Lock stuff */
LOCK_TYPE *InitializeLock(void);
void Lock(LOCK_TYPE *lock);
void Unlock(LOCK_TYPE *lock);

void Clock(unsigned int *time);
