#include <cstdint>
#include <iostream>
#include "helpers.cuh"
#include "encryption.cuh"

void encrypt_cpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters, bool parallel=true) {

    #pragma omp parallel for if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        data[entry] = permute64(entry, num_iters);
}

__global__ 
void decrypt_gpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters) {

    const uint64_t thrdID = blockIdx.x*blockDim.x+threadIdx.x;
    const uint64_t stride = blockDim.x*gridDim.x;

    for (uint64_t entry = thrdID; entry < num_entries; entry += stride)
        data[entry] = unpermute64(data[entry], num_iters);
}

bool check_result_cpu(uint64_t * data, uint64_t num_entries,
                      bool parallel=true) {

    uint64_t counter = 0;

    #pragma omp parallel for reduction(+: counter) if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        counter += data[entry] == entry;

    return counter == num_entries;
}

int main (int argc, char * argv[]) {

    const char * encrypted_file = "/dli/task/encrypted";

    Timer timer;

    const uint64_t num_entries = 1UL << 26;
    const uint64_t num_iters = 1UL << 10;
    const bool openmp = true;

    uint64_t * data_cpu, * data_gpu;
    cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);
    cudaMalloc    (&data_gpu, sizeof(uint64_t)*num_entries);
    check_last_error();

    if (!encrypted_file_exists(encrypted_file)) {
        encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
        write_encrypted_to_file(encrypted_file, data_cpu, sizeof(uint64_t)*num_entries);
    } else {
        read_encrypted_from_file(encrypted_file, data_cpu, sizeof(uint64_t)*num_entries);
    }

    timer.start();
    //cudaMemcpy(data_gpu, data_cpu, 
    //           sizeof(uint64_t)*num_entries, cudaMemcpyHostToDevice);
    const uint64_t num_streams = 32;
    cudaStream_t streams[num_streams];
    for (uint64_t stream = 0; stream < num_streams; stream++)
        cudaStreamCreate(&streams[stream]);
    const uint64_t chunk_size = sdiv(num_entries, num_streams);
    check_last_error();
    
    for (uint64_t stream = 0; stream < num_streams; stream++)
    {
        const uint64_t lower = chunk_size*stream;    
        const uint64_t upper = min(lower+chunk_size, num_entries);
        const uint64_t width = upper-lower;
        cudaMemcpyAsync(data_gpu+lower, data_cpu+lower, sizeof(uint64_t)*width, cudaMemcpyHostToDevice, streams[stream]);
        decrypt_gpu<<<80*32, 64, 0, streams[stream]>>>(data_gpu+lower, width, num_iters);
        cudaMemcpyAsync(data_cpu+lower, data_gpu+lower, sizeof(uint64_t)*width, cudaMemcpyDeviceToHost, streams[stream]);
    }
    //decrypt_gpu<<<80*32, 64>>>(data_gpu, num_entries, num_iters);
    check_last_error();
    //cudaMemcpy(data_cpu, data_gpu, sizeof(uint64_t)*num_entries, cudaMemcpyDeviceToHost);
    timer.stop("total time on GPU");
    check_last_error();

    const bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " 
              << ( success ? "passed" : "failed")
              << std::endl;

    cudaFreeHost(data_cpu);
    cudaFree    (data_gpu);
    check_last_error();
}
