TARGET = OCEAN
OBJS = main.o tlib.o jacobcalc.o jacobcalc2.o laplacalc.o \
	slave1.o slave2.o multi.o linkup.o subblock.o 
CFLAGS = -O2 -OPT:Olimit=2000 -w -DTHREAD_BUILD
LDFLAGS = -lpthread -lm

$(TARGET): $(OBJS)
	cc -n32 $(OBJS) -o $(TARGET) $(LDFLAGS)

.SUFFIXES:
.SUFFIXES:	.o .c .h

.c.o: 
	cc -n32 -c $(CFLAGS) $*.c

jacobcalc.c: decs.h tlib.h extenv.h
linkup.c: decs.h tlib.h extenv.h
slave1.c: decs.h tlib.h extenv.h
jacobcalc2.c: decs.h tlib.h extenv.h
main.c: decs.h tlib.h mainenv.h
slave2.c: decs.h tlib.h extenv.h
laplacalc.c: decs.h tlib.h extenv.h
multi.c: decs.h tlib.h extenv.h
subblock.c: decs.h tlib.h extenv.h

