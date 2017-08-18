#include <stdlib.h>
#include <stdio.h>
#include "tlib.h"

/*
EXTERN_ENV
*/

#include "extenv.h"

void MainInit(void)
{
    pthread_mutex_init(&threadMutex, NULL);
}

void MainEnd(void)
{
    pthread_mutex_destroy(&threadMutex);
    free(parmacsThreads);
}

void Create(int numThreads, void *(*startRoutine)(void *))
{

    /* Lock to access global thread stuff */
    pthread_mutex_lock(&threadMutex);
    threadCount++;

    if (threadCount == 1)
    {
        parmacsThreads = (pthread_t *)malloc(numThreads * sizeof(pthread_t *));
    }
    
    pthread_create(&parmacsThreads[threadCount - 1],
        NULL, startRoutine, (void *)NULL);

    pthread_mutex_unlock(&threadMutex);
}

void JoinAllThreads(void)
{
    int i;
    
    /* Lock to access global thread stuff */
    pthread_mutex_lock(&threadMutex);

    /* join all the threads back after they are done */
    for(i = 0; i < threadCount; i++)
    {
        /* Join threads back */
        pthread_join(parmacsThreads[i], NULL);
    }

    threadCount = 0;
    pthread_mutex_unlock(&threadMutex);
}

void BarrierInit(BARRIER_TYPE *barrier)
{
    barrier->waiters = 0;
    barrier->episode = 0;
    pthread_mutex_init(&barrier->mutex, NULL);
    pthread_cond_init(&barrier->cond, NULL);
}

void BarrierDestroy(BARRIER_TYPE *barrier)
{
    pthread_mutex_destroy(&barrier->mutex);
    pthread_cond_destroy(&barrier->cond);
}

void BarrierReset(BARRIER_TYPE *barrier)
{
    pthread_mutex_lock(&barrier->mutex);

    if (barrier->waiters > 0)
    {
        /* Some theads are waiting, can't init */
        pthread_mutex_unlock(&barrier->mutex);
        perror("Can't rest barrer while in use");
    }

    /* reset episode number */
    barrier->episode = 0;
    pthread_mutex_unlock(&barrier->mutex);
}

void Barrier(BARRIER_TYPE *barrier, int count)
{
    unsigned int thisEpisode;

    /* decrement barrier count under mutex protection */
    pthread_mutex_lock(&barrier->mutex);

    /* get the thread's current episode */
    thisEpisode = barrier->episode;
    ++barrier->waiters;

    if (barrier->waiters == count)
    {
        /******************************************************************
        * Last thread, change episode for next iteration so that threads
        * that now enter this routine to block even if all threads from
        * the previous iteration have not left the routine.
        ******************************************************************/
        ++barrier->episode;
        barrier->waiters = 0;
        pthread_mutex_unlock(&barrier->mutex);
        pthread_cond_broadcast(&barrier->cond);
    }
    else
    {
        while (barrier->episode == thisEpisode)
        {
            /* wait for broadcast on current episode (not any other) */
            pthread_cond_wait(&barrier->cond, &barrier->mutex);
        }

        /* unlock mutex before leaving the routine */
        pthread_mutex_unlock(&barrier->mutex);
    }

}

void Clock(unsigned int *time)
{
    struct timeval fullTime;

    gettimeofday(&fullTime, NULL);
    *time = (unsigned int)((fullTime.tv_usec) +
        1000000 * (fullTime.tv_sec));
}
