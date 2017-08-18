#include <pthread.h>
#include <sys/time.h>

/* Replacements for some of the macros used by parmacs */

/* Won't work with CPP #include must manually edit
#define MAIN_ENV        #include "mainenv.h"
*/
#define MAIN_ENV
#define MAIN_INITENV(X, Y)    MainInit();
#define MAIN_END        MainEnd();

/* Won't work with CPP #include must manually edit
#define EXTERN_ENV      #include "extenv.h"
*/
#define EXTERN_ENV

#define CREATE(X)       Create((nprocs - 1), &X);
#define WAIT_FOR_END(X) JoinAllThreads();

#define LOCKDEC(X)      pthread_mutex_t X;
#define LOCKINIT(X)     pthread_mutex_init(&X, NULL);;
#define LOCK(X)         pthread_mutex_lock(&X);
#define UNLOCK(X)       pthread_mutex_unlock(&X);

#define BARDEC(X)       BARRIER_TYPE X;
#define BARINIT(X)      BarrierInit(&X);
#define BARRIER(X, Y)   Barrier(&X, Y);

#define G_MALLOC(X)     malloc(X);
#define CLOCK(X)        Clock(&X);

typedef struct
{
    int waiters;
    unsigned int episode;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} BARRIER_TYPE;

void MainInit(void);
void MainEnd(void);

void Create(int numThreads, void *(*startRoutine)(void *));
void JoinAllThreads(void);

void BarrierInit(BARRIER_TYPE *barrier);
void BarrierDestroy(BARRIER_TYPE *barrier);
void BarrierReset(BARRIER_TYPE *barrier);
void Barrier(BARRIER_TYPE *barrier, int count);

void Clock(unsigned int *time);
