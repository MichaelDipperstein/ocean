#include <stdlib.h>
#include <stdio.h>
#include "mlib.h"

#include "extenv.h"

#include <malloc.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <ulocks.h>
#include <sys/time.h>
#include <sys/sysmp.h>
#include <strings.h>
#include <stdarg.h>
#include "mpi.h"

/**************************************************************************
*                           Function Prototypes
**************************************************************************/
void ExitError(int rank, int code,
    char *fmt, ... );                           /* Like perror */

void ShmemEnd(void);
void ShmemInit(size_t shmSize, int maxUsers);

/**************************************************************************
*                               Global Variables
**************************************************************************/
static char shmFile[16] = "/tmp/shmXXXXXX";
static int shmId;
static void *shmVAddr;
static void *ap;

#undef MDD_DEBUG

/**************************************************************************
*                                  Functions
**************************************************************************/
void MainInit(int *argc, char **argv[], int y, int *nprocs)
{
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    ShmemInit(y, *nprocs);
}

void MainEnd(void)
{
    ShmemEnd();
    MPI_Finalize();
}

void ShmemInit(size_t shmSize, int maxUsers)
{
    int rank;
	key_t shmKey;
	usptr_t *usPtr;

#ifdef MDD_DEBUG
    _utrace = 1;  /* Trace arena functions */
#endif

    /* Get process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Get ID and create memory pool.  Only 1 PE needs to do this */
    if (rank == 0)
    {
        /* Generate reasonablly unique key */
        shmKey = (key_t)getpid();
        shmKey = (shmKey << 16) | shmKey;
        shmId = shmget(shmKey, shmSize, IPC_CREAT|SHM_R|SHM_W|444);

        if(shmId == -1)
        {
            ExitError(rank, 1, "shmget error");
        }
    }

    /* Broadcast shmId to all PEs */
    MPI_Bcast(&shmId, sizeof(shmId), MPI_BYTE, 0, MPI_COMM_WORLD);

    /**********************************************************************
    * Attach the memory segment to an address.  The rank 0 process will
    * request any address, then broadcast it to all other processes so
    * that they can attach at the same address.
    **********************************************************************/
    if (rank == 0)
    {
        if((shmVAddr = shmat(shmId, (void *)0, 0)) == (void *)(-1))
        {
            ExitError(rank, 2, "shmat error");
        }
    }

    MPI_Bcast(&shmVAddr, sizeof(shmVAddr), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* Attach non-zero processes to address used by zero porcess */
    if (rank != 0)
    {
        if((shmVAddr=shmat(shmId, (void *)shmVAddr, 0)) == (void *)(-1))
        {
            ExitError(rank, 2, "shmat error");
        }
    }

    /**********************************************************************
    * Define an "arena" for shared memory location so that arena functions
    * (amalloc, afree, ..)may be used.  All processes must be linked to the
    * arena, but only process 0 must is required to keep an area pointer.
    ***********************************************************************/
    if (rank == 0)
    {
        mktemp(shmFile);                /* Make file name shared arena */
    }

    /* Broadcast file name to everyone */
    MPI_Bcast(shmFile, strlen(shmFile) + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    usconfig(CONF_INITUSERS, maxUsers); /* Number of arena users */
    MPI_Barrier(MPI_COMM_WORLD);

    if ((usPtr = usinit(shmFile)) == NULL)
    {
        ExitError(rank, 3, "usinit error");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        ap = acreate(shmVAddr, shmSize, MEM_SHARED, usPtr, NULL);
        if (ap == NULL)
        {
            ExitError(rank, 5, "acreate error");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
#ifdef MDD_DEBUG
    printf("process %d shmId 0x%x\n", rank, shmId);
#endif
}

void *GlobalMalloc(size_t size)
{
    int rank;
    static void *addr;

    /* Get process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        addr = amalloc(size, ap);
    }

    /* Broadcast allocation to all processes */
    MPI_Bcast(&addr, sizeof(addr), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (addr == NULL)
    {
        ExitError(5, rank, "amalloc error\n");
    }

    return(addr);
}

void GlobalFree(void *ptr)
{
    int rank;

    /* Get process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        afree(ptr, ap);
    }

    ptr = NULL;
}

void ShmemEnd(void)
{
    int rank;

    /* Get process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        adelete(ap);
        shmctl(shmId, IPC_RMID);
    }

    unlink(shmFile);
}

LOCK_TYPE *InitializeLock(void)
{
    int rank;
    LOCK_TYPE *lock;

    lock = (LOCK_TYPE *)G_MALLOC(sizeof(LOCK_TYPE))

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Initialize lock values */
    if (rank == 0)
    {
        lock->nextAvailable = 0;
        lock->nowServing = 1;
    }

    /* Prevent process from running off with uninitialized locks */
    MPI_Barrier(MPI_COMM_WORLD);
    return(lock);
}

void Lock(LOCK_TYPE *lock)
{
    int myTicket;

    /* Atomically increment next available and save result */
    myTicket = __add_and_fetch(&(lock->nextAvailable), 1);

    /* Loop until nowServing == myTicket */
    while(myTicket != lock->nowServing);
}

void Unlock(LOCK_TYPE *lock)
{
    /* Increment now serving and be done */
    __add_and_fetch(&(lock->nowServing), 1);
}

void Clock(unsigned int *time)
{
    struct timeval fullTime;

    gettimeofday(&fullTime, NULL);
    *time = (unsigned int)((fullTime.tv_usec) +
        1000000 * (fullTime.tv_sec));
}

/**************************************************************************
*   Function   : ExitError
*   Description: If this function is called from process 0, a formatted
*                error message will be displayed.  This process will end
*                MPI communication and exit the program regardless of
*                which process calls it.
*   Parameters : rrank - process rank
*                code - value passed to exit()
*                *fmt - the formatted string to be displayed.
*   Effects    : Displays error messages, ends MPI and exits program.
*   Returned   : None
**************************************************************************/
void ExitError(int rank, int code, char *fmt, ... )
{
    va_list argptr;             /* Argument list pointer */
    char str[129];              /* Formatted message */

    /* Resolve string formatting and store it in str */
    va_start(argptr, fmt);
    vsprintf(str, fmt, argptr);
    va_end(argptr);

    /* Display string */
    printf("Process: %d :: %s", rank, str);

    MPI_Finalize();
    exit(code);
}
