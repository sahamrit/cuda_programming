#include <bits/stdc++.h>
#include "helpers.cuh"
#include "encryption.cuh"

using namespace std;

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

    int num_gpus ;
    cudaGetDeviceCount(&num_gpus);
    
    uint64_t * data_cpu;    
    uint64_t  *data_gpus[num_gpus];

    uint64_t segment_entries = sdiv(num_entries , num_gpus);

    cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);

    // For each GPU...
    for (int i = 0 ; i < num_gpus; i++){
        cudaSetDevice(i);

        uint64_t lower_idx = segment_entries * i;
        uint64_t curr_segment_entries = min (num_entries - lower_idx, segment_entries);
        size_t curr_segment_size = curr_segment_entries * sizeof(uint64_t);         
        cudaMalloc (&data_gpus[i], curr_segment_size);   
    }
    
    check_last_error();

    if (!encrypted_file_exists(encrypted_file)) {
        encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
        write_encrypted_to_file(encrypted_file, data_cpu, sizeof(uint64_t)*num_entries);
    } else {
        read_encrypted_from_file(encrypted_file, data_cpu, sizeof(uint64_t)*num_entries);
    }

    // For each GPU...
    for (int i = 0 ; i < num_gpus; i++){
        cudaSetDevice(i);

        uint64_t lower_idx = segment_entries * i;
        uint64_t curr_segment_entries = min (num_entries - lower_idx, segment_entries);
        size_t curr_segment_size = curr_segment_entries * sizeof(uint64_t); 

        cudaMemcpy(data_gpus[i], &data_cpu[lower_idx], curr_segment_size, cudaMemcpyHostToDevice );      
    }

    check_last_error();

    timer.start();

    // For each GPU...
    for (int i = 0 ; i < num_gpus; i++){
        cudaSetDevice(i);

        uint64_t lower_idx = segment_entries * i;
        uint64_t curr_segment_entries = min (num_entries - lower_idx, segment_entries);

        decrypt_gpu<<<80*32, 64>>>(data_gpus[i], curr_segment_entries, num_iters);  
    }
    // As you refactor, be sure to stop the timer after all GPU kernel launches are complete
    timer.stop("total kernel execution on GPUs");
    check_last_error();

    // For each GPU...
    for (int i = 0 ; i < num_gpus; i++){
        cudaSetDevice(i);

        uint64_t lower_idx = segment_entries * i;
        uint64_t curr_segment_entries = min (num_entries - lower_idx, segment_entries);
        size_t curr_segment_size = curr_segment_entries * sizeof(uint64_t); 

        cudaMemcpy(&data_cpu[lower_idx], data_gpus[i], curr_segment_size, cudaMemcpyDeviceToHost );      
    }

    check_last_error();

    const bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " 
              << ( success ? "passed" : "failed")
              << std::endl;

    cudaFreeHost(data_cpu);

    // For each GPU...
    for (int i = 0 ; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaFree(data_gpus[i]);   
    }
    check_last_error();
}
