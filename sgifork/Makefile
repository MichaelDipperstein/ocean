TARGET = OCEAN
OBJS = main.o jacobcalc.o jacobcalc2.o laplacalc.o \
	slave1.o slave2.o multi.o linkup.o subblock.o \
	shmem.o
CFLAGS = -n32 -O2
LDFLAGS = -lm
MACROS = ./c.m4.sgi

x = *

$(TARGET): $(OBJS)
	cc -n32 $(OBJS) -o $(TARGET) $(LDFLAGS)

.SUFFIXES:
.SUFFIXES:	.o .c .C .h .H

.H.h: 
	m4 ${MACROS} $*.H > $*.h

.C.c: 
	m4 $(MACROS) $*.C > $*.c

.c.o: 
	cc -c $(CFLAGS) $*.c

.C.o: 
	m4 $(MACROS) $*.C > $*.c
	cc -c $(CFLAGS) $*.c

decs.h: decs.H
jacobcalc.c: decs.h
linkup.c: decs.h
slave1.c: decs.h
jacobcalc2.c: decs.h
main.c: decs.h
slave2.c: decs.h
laplacalc.c: decs.h
multi.c: decs.h
subblock.c: decs.h

