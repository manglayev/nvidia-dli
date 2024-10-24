# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.

CC := nvc++
ACCFLAGS := -ta=tesla:managed,lineinfo -Minfo=accel

rdf: rdf.cpp
	${CC} ${CFLAGS} ${ACCFLAGS} -o rdf rdf.cpp -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/lib64 -lnvToolsExt

clean:
	rm -f *.o rdf
