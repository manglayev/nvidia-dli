CUDACXX=nvcc
CUDACXXFLAGS=-arch=sm_70 -O3
CXXFLAGS=-march=native -fopenmp
NSYS=nsys profile
NSYSFLAGS=--stats=true --force-overwrite=true

all: mgpu_stream

mgpu_stream: mgpu_stream.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" mgpu_stream.cu -o mgpu_stream

mgpu_stream_solution: mgpu_stream_solution.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" mgpu_stream_solution.cu -o mgpu_stream_solution

profile: mgpu_stream
	$(NSYS) $(NSYSFLAGS) -o mgpu-stream-report ./mgpu_stream

profile_solution: mgpu_stream_solution
	$(NSYS) $(NSYSFLAGS) -o mgpu-stream-solution-report ./mgpu_stream_solution

clean:
	rm -f mgpu_stream mgpu_stream_solution *.qdrep *.sqlite
