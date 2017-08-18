/*************************************************************************/
/*                                                                       */
/*  Copyright (c) 1994 Stanford University                               */
/*                                                                       */
/*  All rights reserved.                                                 */
/*                                                                       */
/*  Permission is given to use, copy, and modify this software for any   */
/*  non-commercial purpose as long as this copyright notice is not       */
/*  removed.  All other uses, including redistribution in whole or in    */
/*  part, are forbidden without prior written permission.                */
/*                                                                       */
/*  This software is provided with absolutely no warranty and no         */
/*  support.                                                             */
/*                                                                       */
/*************************************************************************/

/*************************************************************************/
/*                                                                       */
/*  SPLASH Ocean Code                                                    */
/*                                                                       */
/*  This application studies the role of eddy and boundary currents in   */
/*  influencing large-scale ocean movements.  This implementation uses   */
/*  dynamically allocated four-dimensional arrays for grid data storage. */
/*                                                                       */
/*  Command line options:                                                */
/*                                                                       */
/*     -nN : Simulate NxN ocean.  N must be (power of 2)+2.              */
/*     -pP : P = number of processors.  P must be power of 2.            */
/*     -eE : E = error tolerance for iterative relaxation.               */
/*     -rR : R = distance between grid points in meters.                 */
/*     -tT : T = timestep in seconds.                                    */
/*     -s  : Print timing statistics.                                    */
/*     -o  : Print out relaxation residual values.                       */
/*     -h  : Print out command line options.                             */
/*                                                                       */
/*  Default: OCEAN -n130 -p1 -e1e-7 -r20000.0 -t28800.0                  */
/*                                                                       */
/*  NOTE: This code works under both the FORK and SPROC models.          */
/*                                                                       */
/*************************************************************************/
#include "gdefs.h"

#define DEFAULT_N      258
#define DEFAULT_P        1
#define DEFAULT_E        1e-7
#define DEFAULT_T    28800.0
#define DEFAULT_R    20000.0
#define UP               0
#define DOWN             1
#define LEFT             2
#define RIGHT            3
#define UPLEFT           4
#define UPRIGHT          5
#define DOWNLEFT         6
#define DOWNRIGHT        7
#define PAGE_SIZE     4096

#define d_size  *g_d_size

#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef THREAD_BUILD
#include "tlib.h"
#endif

#ifdef MPI_BUILD
#include "mlib.h"
#endif

#include "mainenv.h"

struct multi_struct {
   double err_multi;
} *multi;

struct global_struct {
   int id;
   int starttime;
   int trackstart;
   double psiai;
   double psibi;
} *global;

double ****psi;
double ****psim;
double ***psium;
double ***psilm;
double ***psib;
double ***ga;
double ***gb;
double ****work1;
double ***work2;
double ***work3;
double ****work4;
double ****work5;
double ***work6;
double ****work7;
double ****temparray;
double ***tauz;
double ***oldga;
double ***oldgb;
double *f;
double ****q_multi;
double ****rhs_multi;

struct locks_struct {
   LOCKDEC(idlock)
   LOCKDEC(psiailock)
   LOCKDEC(psibilock)
   LOCKDEC(donelock)
   LOCKDEC(error_lock)
   LOCKDEC(bar_lock)
} *locks;

struct bars_struct {
   BARDEC(iteration)
   BARDEC(gsudn)
   BARDEC(p_setup) 
   BARDEC(p_redph) 
   BARDEC(p_soln) 
   BARDEC(p_subph) 
   BARDEC(sl_prini)
   BARDEC(sl_psini)
   BARDEC(sl_onetime)
   BARDEC(sl_phase_1)
   BARDEC(sl_phase_2)
   BARDEC(sl_phase_3)
   BARDEC(sl_phase_4)
   BARDEC(sl_phase_5)
   BARDEC(sl_phase_6)
   BARDEC(sl_phase_7)
   BARDEC(sl_phase_8)
   BARDEC(sl_phase_9)
   BARDEC(sl_phase_10)
   BARDEC(error_barrier)
} *bars;

void subblock();
void slave();
int log_2(int);
void printerr(char *);
void InitGlobals(void);

int nprocs;
double h1;
double h3;
double h;
double lf;
double res;
double dtau;
double f0;
double beta;
double gpr;
int im;
int jm;
double tolerance;
double eig2;
double ysca;
int jmm1;
double pi;
double t0;
double outday0;
double outday1;
double outday2;
double outday3;
double factjacob;
double factlap;
int numlev;
int *imx;
int *jmx;
double *lev_res;
double *lev_tol;
double maxwork;

struct Global_Private {
  char pad[PAGE_SIZE];
  int *rel_num_x;
  int *rel_num_y;
  int *eist;     
  int *ejst;     
  int *oist;     
  int *ojst;     
  int *rlist;    
  int *rljst;    
  int *rlien;    
  int *rljen;    
  int rownum;
  int colnum;
  int neighbors[8];
  double multi_time;
  double total_time;
} *gp;

double *i_int_coeff;
double *j_int_coeff;
int xprocs;
int yprocs;
int *xpts_per_proc;
int *ypts_per_proc;
int minlevel;
int do_stats;
int do_output;

void main(argc, argv)

int argc;
char *argv[];

{
   int i;
   int j;
   int k;
   double work_multi;
   int my_num;
   int x_part;
   int y_part;
   int *g_d_size;
   int itemp;
   int jtemp;
   double procsqrt;
   FILE *fileptr;
   int iindex;
   int temp = 0;
   char c;
   double min_total;
   double max_total;
   double avg_total;
   double min_multi;
   double max_multi;
   double avg_multi;
   double min_frac;
   double max_frac;
   double avg_frac;
   int ch;
   extern char *optarg;
   unsigned int computeend;
   unsigned int start;

   MAIN_INITENV(,60000000) 

   MPI_Barrier(MPI_COMM_WORLD);
   g_d_size = (int *)G_MALLOC(sizeof(int));
  
   printf("Process %d about to init globals\n", myRank);
   
   MPI_Barrier(MPI_COMM_WORLD);
   InitGlobals();

if (myRank == 0)
{
   h1 = 1000.0;
   h3 = 4000.0;
   h = 5000.0;
   lf = -5.12e11;
   res = DEFAULT_R;
   dtau = DEFAULT_T;
   f0 = 8.3e-5;
   beta = 2.0e-11;
   gpr = 0.02;
   im = DEFAULT_N;
   tolerance = DEFAULT_E;
   t0 = 0.5e-4 ;
   outday0 = 1.0;
   outday1 = 2.0;
   outday2 = 2.0;
   outday3 = 2.0;
   maxwork = 10000.0;

   CLOCK(start)

   while ((ch = getopt(argc, argv, "n:p:e:r:t:soh")) != -1) {
     switch(ch) {
     case 'n': im = atoi(optarg);
               if (log_2(im-2) == -1) {
                 printerr("Grid must be ((power of 2)+2) in each dimension\n");
                 exit(-1);
               }
               break;
     case 'p': nprocs = atoi(optarg);
               if (nprocs < 1) {
                 printerr("P must be >= 1\n");
                 exit(-1);
               }
               if (log_2(nprocs) == -1) {
                 printerr("P must be a power of 2\n");
                 exit(-1);
               }
               break;
     case 'e': tolerance = atof(optarg); break;
     case 'r': res = atof(optarg); break;
     case 't': dtau = atof(optarg); break;
     case 's': do_stats = !do_stats; break;
     case 'o': do_output = !do_output; break;
     case 'h': printf("Usage: OCEAN <options>\n\n");
               printf("options:\n");
               printf("  -nN : Simulate NxN ocean.  N must be (power of 2)+2.\n");
               printf("  -pP : P = number of processors.  P must be power of 2.\n");
               printf("  -eE : E = error tolerance for iterative relaxation.\n");
               printf("  -rR : R = distance between grid points in meters.\n");
               printf("  -tT : T = timestep in seconds.\n");
               printf("  -s  : Print timing statistics.\n");
               printf("  -o  : Print out relaxation residual values.\n");
               printf("  -h  : Print out command line options.\n\n");
               printf("Default: OCEAN -n%1d -p%1d -e%1g -r%1g -t%1g\n",
                       DEFAULT_N,DEFAULT_P,DEFAULT_E,DEFAULT_R,DEFAULT_T);
               exit(0);
               break;
     }
   }

   jm = im;
   printf("\n");
   printf("Ocean simulation with W-cycle multigrid solver\n");
   printf("    Processors                         : %1d\n",nprocs);
   printf("    Grid size                          : %1d x %1d\n",im,jm);
   printf("    Grid resolution (meters)           : %0.2f\n",res);
   printf("    Time between relaxations (seconds) : %0.0f\n",dtau);
   printf("    Error tolerance                    : %0.7g\n",tolerance);
   printf("\n");

   xprocs = 0;
   yprocs = 0;
   procsqrt = sqrt((double) nprocs);
   j = (int) procsqrt;
   while ((xprocs == 0) && (j > 0)) {
     k = nprocs / j;
     if (k * j == nprocs) {
       if (k > j) {
         xprocs = j;
         yprocs = k;
       } else {
         xprocs = k;
         yprocs = j;
       }
     }
     j--;
   }
   if (xprocs == 0) {
     printerr("Could not find factors for subblocking\n");
     exit(-1);
   }  

   minlevel = 0;
   itemp = 1;
   jtemp = 1;
   numlev = 0;
   minlevel = 0;
   while (itemp < (im-2)) {
     itemp = itemp*2;
     jtemp = jtemp*2;
     if ((itemp/yprocs > 1) && (jtemp/xprocs > 1)) {
       numlev++;
     }
   }  
   
   if (numlev == 0) {
     printerr("Must have at least 2 grid points per processor in each dimension\n");
     exit(-1);
   }
}

   MPI_Barrier(MPI_COMM_WORLD);     /* Let every process sync up */

   imx = (int *) G_MALLOC(numlev*sizeof(int));
   jmx = (int *) G_MALLOC(numlev*sizeof(int));
   lev_res = (double *) G_MALLOC(numlev*sizeof(double));
   lev_tol = (double *) G_MALLOC(numlev*sizeof(double));
   i_int_coeff = (double *) G_MALLOC(numlev*sizeof(double));
   j_int_coeff = (double *) G_MALLOC(numlev*sizeof(double));
   xpts_per_proc = (int *) G_MALLOC(numlev*sizeof(int));
   ypts_per_proc = (int *) G_MALLOC(numlev*sizeof(int));

if (myRank == 0)
{
   imx[numlev-1] = im;
   jmx[numlev-1] = jm;
   lev_res[numlev-1] = res;
   lev_tol[numlev-1] = tolerance;

   for (i=numlev-2;i>=0;i--) {
     imx[i] = ((imx[i+1] - 2) / 2) + 2;
     jmx[i] = ((jmx[i+1] - 2) / 2) + 2;
     lev_res[i] = lev_res[i+1] * 2;
   }

   for (i=0;i<numlev;i++) {
     xpts_per_proc[i] = (jmx[i]-2) / xprocs;
     ypts_per_proc[i] = (imx[i]-2) / yprocs;
   }  
   for (i=numlev-1;i>=0;i--) {
     if ((xpts_per_proc[i] < 2) || (ypts_per_proc[i] < 2)) {
       minlevel = i+1;
       break;
     }
   }    
 
   for (i=0;i<numlev;i++) {
     temp += imx[i];
   }
   temp = 0;
   j = 0;
   for (k=0;k<numlev;k++) {
     for (i=0;i<imx[k];i++) {
       j++;
       temp += jmx[k];
     }
   }

   d_size = nprocs*sizeof(double ***);
}

   MPI_Barrier(MPI_COMM_WORLD);     /* Let every process sync up */

   psi = (double ****) G_MALLOC(d_size);
   psim = (double ****) G_MALLOC(d_size);
   work1 = (double ****) G_MALLOC(d_size);
   work4 = (double ****) G_MALLOC(d_size);
   work5 = (double ****) G_MALLOC(d_size);
   work7 = (double ****) G_MALLOC(d_size);
   temparray = (double ****) G_MALLOC(d_size);

   d_size = 2*sizeof(double **);
   for (i=0;i<nprocs;i++) {
     psi[i] = (double ***) G_MALLOC(d_size);
     psim[i] = (double ***) G_MALLOC(d_size);
     work1[i] = (double ***) G_MALLOC(d_size);
     work4[i] = (double ***) G_MALLOC(d_size);
     work5[i] = (double ***) G_MALLOC(d_size);
     work7[i] = (double ***) G_MALLOC(d_size);
     temparray[i] = (double ***) G_MALLOC(d_size);
   }

   d_size = nprocs*sizeof(double **);
   psium = (double ***) G_MALLOC(d_size);
   psilm = (double ***) G_MALLOC(d_size);
   psib = (double ***) G_MALLOC(d_size);
   ga = (double ***) G_MALLOC(d_size);
   gb = (double ***) G_MALLOC(d_size);
   work2 = (double ***) G_MALLOC(d_size);
   work3 = (double ***) G_MALLOC(d_size);
   work6 = (double ***) G_MALLOC(d_size);
   tauz = (double ***) G_MALLOC(d_size);
   oldga = (double ***) G_MALLOC(d_size);
   oldgb = (double ***) G_MALLOC(d_size);

   gp = (struct Global_Private *) G_MALLOC((nprocs+1)*sizeof(struct Global_Private));
   for (i=0;i<nprocs;i++) {
     gp[i].rel_num_x = (int *) G_MALLOC(numlev*sizeof(int));
     gp[i].rel_num_y = (int *) G_MALLOC(numlev*sizeof(int));
     gp[i].eist = (int *) G_MALLOC(numlev*sizeof(int));
     gp[i].ejst = (int *) G_MALLOC(numlev*sizeof(int));
     gp[i].oist = (int *) G_MALLOC(numlev*sizeof(int));
     gp[i].ojst = (int *) G_MALLOC(numlev*sizeof(int));
     gp[i].rlist = (int *) G_MALLOC(numlev*sizeof(int));
     gp[i].rljst = (int *) G_MALLOC(numlev*sizeof(int));
     gp[i].rlien = (int *) G_MALLOC(numlev*sizeof(int));
     gp[i].rljen = (int *) G_MALLOC(numlev*sizeof(int));
     gp[i].multi_time = 0;
     gp[i].total_time = 0;
   }

if (myRank == 0)
{
   subblock();

   x_part = (jm - 2)/xprocs + 2;
   y_part = (im - 2)/yprocs + 2;

   d_size = x_part*y_part*sizeof(double) + y_part*sizeof(double *);
}

   MPI_Barrier(MPI_COMM_WORLD);
   
   global = (struct global_struct *) G_MALLOC(sizeof(struct global_struct));  
   for (i=0;i<nprocs;i++) {
     psi[i][0] = (double **) G_MALLOC(d_size);
     psi[i][1] = (double **) G_MALLOC(d_size);
     psim[i][0] = (double **) G_MALLOC(d_size);
     psim[i][1] = (double **) G_MALLOC(d_size);
     psium[i] = (double **) G_MALLOC(d_size);
     psilm[i] = (double **) G_MALLOC(d_size);
     psib[i] = (double **) G_MALLOC(d_size);
     ga[i] = (double **) G_MALLOC(d_size);
     gb[i] = (double **) G_MALLOC(d_size);
     work1[i][0] = (double **) G_MALLOC(d_size);
     work1[i][1] = (double **) G_MALLOC(d_size);
     work2[i] = (double **) G_MALLOC(d_size);
     work3[i] = (double **) G_MALLOC(d_size);
     work4[i][0] = (double **) G_MALLOC(d_size);
     work4[i][1] = (double **) G_MALLOC(d_size);
     work5[i][0] = (double **) G_MALLOC(d_size);
     work5[i][1] = (double **) G_MALLOC(d_size);
     work6[i] = (double **) G_MALLOC(d_size);
     work7[i][0] = (double **) G_MALLOC(d_size);
     work7[i][1] = (double **) G_MALLOC(d_size);
     temparray[i][0] = (double **) G_MALLOC(d_size);
     temparray[i][1] = (double **) G_MALLOC(d_size);
     tauz[i] = (double **) G_MALLOC(d_size);
     oldga[i] = (double **) G_MALLOC(d_size);
     oldgb[i] = (double **) G_MALLOC(d_size);
   }
   f = (double *) G_MALLOC(im*sizeof(double));

   multi = (struct multi_struct *) G_MALLOC(sizeof(struct multi_struct));


if (myRank == 0)
{
   d_size = numlev*sizeof(double **);
   if (numlev%2 == 1) {         /* To make sure that the actual data 
                                   starts double word aligned, add an extra
                                   pointer */
     d_size += sizeof(double **);
   }
   for (i=0;i<numlev;i++) {
     d_size += ((imx[i]-2)/yprocs+2)*((jmx[i]-2)/xprocs+2)*sizeof(double)+
              ((imx[i]-2)/yprocs+2)*sizeof(double *);
   }

   d_size *= nprocs;

   if (nprocs%2 == 1) {         /* To make sure that the actual data 
                                   starts double word aligned, add an extra
                                   pointer */
     d_size += sizeof(double ***);
   }

   d_size += nprocs*sizeof(double ***);
}

   MPI_Barrier(MPI_COMM_WORLD);

   q_multi = (double ****) G_MALLOC(d_size);
   rhs_multi = (double ****) G_MALLOC(d_size);

   locks = (struct locks_struct *) G_MALLOC(sizeof(struct locks_struct));
   bars = (struct bars_struct *) G_MALLOC(sizeof(struct bars_struct));

   LOCKINIT(locks->idlock)
   LOCKINIT(locks->psiailock)
   LOCKINIT(locks->psibilock)
   LOCKINIT(locks->donelock)
   LOCKINIT(locks->error_lock)
   LOCKINIT(locks->bar_lock)

   BARINIT(bars->iteration)
   BARINIT(bars->gsudn)
   BARINIT(bars->p_setup) 
   BARINIT(bars->p_redph) 
   BARINIT(bars->p_soln) 
   BARINIT(bars->p_subph) 
   BARINIT(bars->sl_prini)
   BARINIT(bars->sl_psini)
   BARINIT(bars->sl_onetime)
   BARINIT(bars->sl_phase_1)
   BARINIT(bars->sl_phase_2)
   BARINIT(bars->sl_phase_3)
   BARINIT(bars->sl_phase_4)
   BARINIT(bars->sl_phase_5)
   BARINIT(bars->sl_phase_6)
   BARINIT(bars->sl_phase_7)
   BARINIT(bars->sl_phase_8)
   BARINIT(bars->sl_phase_9)
   BARINIT(bars->sl_phase_10)
   BARINIT(bars->error_barrier)

if (myRank == 0)
{
   link_all();

   multi->err_multi = 0.0;
   i_int_coeff[0] = 0.0;
   j_int_coeff[0] = 0.0;
   for (i=0;i<numlev;i++) {
     i_int_coeff[i] = 1.0/(imx[i]-1);
     j_int_coeff[i] = 1.0/(jmx[i]-1);
   }

/* initialize constants and variables

   id is a global shared variable that has fetch-and-add operations
   performed on it by processes to obtain their pids.   */

   global->id = 0;
   global->psibi = 0.0;
   pi = atan(1.0);
   pi = 4.*pi;

   factjacob = -1./(12.*res*res);
   factlap = 1./(res*res);
   eig2 = -h*f0*f0/(h1*h3*gpr);

   jmm1 = jm-1 ;
   ysca = ((double) jmm1)*res ;

   im = (imx[numlev-1]-2)/yprocs + 2;
   jm = (jmx[numlev-1]-2)/xprocs + 2;
}

#if 0
   for (i=1;i<nprocs;i++) {
     CREATE(slave)  
   }
#endif

if (myRank == 0)
{
   if (do_output) {
     printf("                       MULTIGRID OUTPUTS\n");
   }
}

   slave();
   WAIT_FOR_END(nprocs-1)

if (myRank == 0)
{
   CLOCK(computeend)

   printf("\n");
   printf("                       PROCESS STATISTICS\n");
   printf("                  Total          Multigrid         Multigrid\n");
   printf(" Proc             Time             Time            Fraction\n");
   printf("    0   %15.0f    %15.0f        %10.3f\n",
          gp[0].total_time,gp[0].multi_time,
          gp[0].multi_time/gp[0].total_time);

   if (do_stats) {
     min_total = max_total = avg_total = gp[0].total_time;
     min_multi = max_multi = avg_multi = gp[0].multi_time;
     min_frac = max_frac = avg_frac = gp[0].multi_time/gp[0].total_time;
     for (i=1;i<nprocs;i++) {
       if (gp[i].total_time > max_total) {
         max_total = gp[i].total_time;
       }
       if (gp[i].total_time < min_total) {
         min_total = gp[i].total_time;
       }
       if (gp[i].multi_time > max_multi) {
         max_multi = gp[i].multi_time;
       }
       if (gp[i].multi_time < min_multi) {
         min_multi = gp[i].multi_time;
       }
       if (gp[i].multi_time/gp[i].total_time > max_frac) {
         max_frac = gp[i].multi_time/gp[i].total_time;
       }
       if (gp[i].multi_time/gp[i].total_time < min_frac) {
         min_frac = gp[i].multi_time/gp[i].total_time;
       }
       avg_total += gp[i].total_time;
       avg_multi += gp[i].multi_time;
       avg_frac += gp[i].multi_time/gp[i].total_time;
     }
     avg_total = avg_total / nprocs;
     avg_multi = avg_multi / nprocs;
     avg_frac = avg_frac / nprocs;
     for (i=1;i<nprocs;i++) {
       printf("  %3d   %15.0f    %15.0f        %10.3f\n",
	      i,gp[i].total_time,gp[i].multi_time,
	      gp[i].multi_time/gp[i].total_time);
     }
     printf("  Avg   %15.0f    %15.0f        %10.3f\n",
            avg_total,avg_multi,avg_frac);
     printf("  Min   %15.0f    %15.0f        %10.3f\n",
            min_total,min_multi,min_frac);
     printf("  Max   %15.0f    %15.0f        %10.3f\n",
            max_total,max_multi,max_frac);
   }
   printf("\n");

   global->starttime = start;
   printf("                       TIMING INFORMATION\n");
   printf("Start time                        : %16d\n",
           global->starttime);
   printf("Initialization finish time        : %16d\n",
           global->trackstart);
   printf("Overall finish time               : %16d\n",
           computeend);
   printf("Total time with initialization    : %16d\n",
           computeend-global->starttime);
   printf("Total time without initialization : %16d\n",
           computeend-global->trackstart);
   printf("    (excludes first timestep)\n");
   printf("\n");

}   /* end if (myRank == 0) */

   MAIN_END
}

int log_2(number)

int number;

{
  int cumulative = 1;
  int out = 0;
  int done = 0;

  while ((cumulative < number) && (!done) && (out < 50)) {
    if (cumulative == number) {
      done = 1;
    } else {
      cumulative = cumulative * 2;
      out ++;
    }
  }

  if (cumulative == number) {
    return(out);
  } else {
    return(-1);
  }
}

void printerr(s)

char *s;

{
  fprintf(stderr,"ERROR: %s\n",s);
}

void InitGlobals(void)
{
    printf("%d started initializing globals\n", myRank);
 
    g_eig2 = (double *)G_MALLOC(sizeof(double));
    g_ysca = (double *)G_MALLOC(sizeof(double));
    g_jmm1 = (int *)G_MALLOC(sizeof(int));
    g_pi = (double *)G_MALLOC(sizeof(double));
    g_t0 = (double *)G_MALLOC(sizeof(double));

    g_factjacob = (double *)G_MALLOC(sizeof(double));
    g_factlap = (double *)G_MALLOC(sizeof(double));

    g_xprocs = (int *)G_MALLOC(sizeof(int));
    g_yprocs = (int *)G_MALLOC(sizeof(int));

    g_numlev = (int *)G_MALLOC(sizeof(int));
    g_maxwork = (double *)G_MALLOC(sizeof(double));
    g_minlevel = (int *)G_MALLOC(sizeof(int));
    g_outday0 = (double *)G_MALLOC(sizeof(double));
    g_outday1 = (double *)G_MALLOC(sizeof(double));
    g_outday2 = (double *)G_MALLOC(sizeof(double));
    g_outday3 = (double *)G_MALLOC(sizeof(double));

    g_h1 = (double *)G_MALLOC(sizeof(double));
    g_h3 = (double *)G_MALLOC(sizeof(double));
    g_h = (double *)G_MALLOC(sizeof(double));
    g_lf = (double *)G_MALLOC(sizeof(double));
    g_res = (double *)G_MALLOC(sizeof(double));
    g_dtau = (double *)G_MALLOC(sizeof(double));
    g_f0 = (double *)G_MALLOC(sizeof(double));
    g_beta = (double *)G_MALLOC(sizeof(double));
    g_gpr = (double *)G_MALLOC(sizeof(double));
    g_im = (int *)G_MALLOC(sizeof(int));
    g_jm = (int *)G_MALLOC(sizeof(int));
    g_do_stats = (int *)G_MALLOC(sizeof(int));
    g_do_output = (int *)G_MALLOC(sizeof(int));

    printf("%d Finished initializing globals\n", myRank);
}
