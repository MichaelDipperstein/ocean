TARGET = OCEAN
OBJS = main.o mlib.o jacobcalc.o jacobcalc2.o laplacalc.o \
	slave1.o slave2.o multi.o linkup.o subblock.o 
CFLAGS = -O2 -OPT:Olimit=2000 -w -DMPI_BUILD
LDFLAGS = -lmpi -lm

mtest: mtest.o mlib.o
	cc -n32 $(CFLAGS) mtest.o mlib.o -o mtest $(LDFLAGS)

mtest.o: mtest.c mlib.h
	cc -n32 $(CFLAGS) -c mtest.c -o mtest.o

mlib.o: mlib.c mlib.h
	cc -n32 $(CFLAGS) -c mlib.c -o mlib.o


