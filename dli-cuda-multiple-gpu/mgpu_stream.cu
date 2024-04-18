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

int main (int argc, char * argv[])
{
    const char * encrypted_file = "/dli/task/encrypted";
    Timer timer;
    const uint64_t num_entries = 1UL << 26;
    const uint64_t num_iters = 1UL << 10;
    const bool openmp = true;
    const uint64_t num_streams = 32;
        
    //cudaMalloc(&data_gpu, sizeof(uint64_t)*num_entries);
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    
    const uint64_t stream_chunk_size = sdiv(sdiv(num_entries, num_gpus), num_streams);
    const uint64_t gpu_chunk_size = stream_chunk_size*num_streams;
    
    uint64_t * data_cpu, * data_gpu[num_gpus];
    cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);
    
    cudaStream_t streams[num_gpus][num_streams]; // 2D array containing number of streams for each GPU.
    // For each available GPU...
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++)
    {
        // ...set as active device...
        cudaSetDevice(gpu);
        for (uint64_t stream = 0; stream < num_streams; stream++)
        // ...create and store its number of streams.
        cudaStreamCreate(&streams[gpu][stream]);
    }
    
    
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++)
    {
        cudaSetDevice(gpu);
        const uint64_t lower = gpu_chunk_size*gpu;
        const uint64_t upper = min(lower+gpu_chunk_size, num_entries);
        const uint64_t width = upper-lower;
        cudaMalloc(&data_gpu[gpu], sizeof(uint64_t)*width);        
    }
    check_last_error();
    
    if (!encrypted_file_exists(encrypted_file))
    {
        encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
        write_encrypted_to_file(encrypted_file, data_cpu, sizeof(uint64_t)*num_entries);
    }
    else
    {
        read_encrypted_from_file(encrypted_file, data_cpu, sizeof(uint64_t)*num_entries);
    }
    
    timer.start();
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++)
    {
        cudaSetDevice(gpu);
        // For each stream (on each GPU)...
        for (uint64_t stream = 0; stream < num_streams; stream++)
        {
            // Calculate index offset for this stream's chunk of data within the GPU's chunk of data...
            const uint64_t stream_offset = stream_chunk_size*stream;

            // ...get the lower index within all data, and width of this stream's data chunk...
            const uint64_t lower = gpu_chunk_size*gpu+stream_offset;
            const uint64_t upper = min(lower+stream_chunk_size, num_entries);
            const uint64_t width = upper-lower;

            // ...perform async HtoD memory copy...
            cudaMemcpyAsync(data_gpu[gpu]+stream_offset, // This stream's data within this GPU's data.
                            data_cpu+lower,              // This stream's data within all CPU data.
                            sizeof(uint64_t)*width,      // This stream's chunk size worth of data.
                            cudaMemcpyHostToDevice,
                            streams[gpu][stream]);       // Using this stream for this GPU.

            decrypt_gpu<<<80*32, 64, 0, streams[gpu][stream]>>>(data_gpu[gpu]+stream_offset, width, num_iters);
                                                         // Using this stream for this GPU.
                                                         // This stream's data within this GPU's data.
                                                         // This stream's chunk size worth of data.
            cudaMemcpyAsync(data_cpu+lower,              // This stream's data within all CPU data.
                            data_gpu[gpu]+stream_offset, // This stream's data within this GPU's data.
                            sizeof(uint64_t)*width,
                            cudaMemcpyDeviceToHost,
                            streams[gpu][stream]);       // Using this stream for this GPU.
        }
    }    
    timer.stop("total time on GPU");
    check_last_error();
    /*
        cudaMemcpy(data_gpu, data_cpu, sizeof(uint64_t)*num_entries, cudaMemcpyHostToDevice);
        check_last_error();
        decrypt_gpu<<<80*32, 64>>>(data_gpu, num_entries, num_iters);
        check_last_error();
        cudaMemcpy(data_cpu, data_gpu, sizeof(uint64_t)*num_entries, cudaMemcpyDeviceToHost);
        timer.stop("total time on GPU");
        check_last_error();
    */
    const bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " 
              << ( success ? "passed" : "failed")
              << std::endl;

    cudaFreeHost(data_cpu);
    //cudaFree    (data_gpu);
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++)
    {
        cudaSetDevice(gpu);
        cudaFree(data_gpu[gpu]);
    }
    check_last_error();
}
