#include <cuda_runtime.h>
#include <cuda.h>
#include "gputypes.h"
#include <signal.h>
#include <vector>
#include <stdio.h>
#include <assert.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


#define CU_CONST_MEMORY_SIZE (1 << 16)

__constant__ GPUClause clause_data[(1 << 16) / sizeof(GPUClause)];


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) raise(SIGABRT);
   }
   code = cudaGetLastError();
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert last error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) raise(SIGABRT);
   }
}

__device__ uint64_t get_global_id() {
    // TODO: y and z
    return blockDim.x * blockIdx.x + threadIdx.x;
}


__global__ void solveComponent(
        unsigned* solution_counter,
        unsigned clause_count,
        unsigned scale_factor) 
{
    uint64_t id = get_global_id() << scale_factor;
    unsigned hits = 0;

    for (int assignment_idx = 0; assignment_idx < (1 << scale_factor); assignment_idx++) {
        uint64_t assignment = id | assignment_idx;
        bool conflict = 0;
        for (auto clause_idx = 0; clause_idx < clause_count; clause_idx++) {
            GPUClause clause = clause_data[clause_idx];
            // flip negatively signed variables
            auto signed_assignment = assignment ^ clause.neg_vars;
            // clause is not satisfied -> abort
            if ((signed_assignment & clause.vars) == 0) {
                conflict = 1;
                break;
            }
        }
        if (!conflict) {
            hits++;
        }
    }
    atomicAdd_block(solution_counter, hits);
}

unsigned long long componentModelCount(const std::vector<GPUClause>& clauses, uint64_t variable_count) {
    
    int64_t threadsPerBlock = 1024;
    
    uint64_t assignments = 1l << variable_count;

    // in log2, https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html
    const int64_t max_parallel_threads = 16;
    int64_t scale_factor = max(0l, (long)variable_count - max_parallel_threads);
   
    uint64_t threads = assignments >> scale_factor;
    int64_t blocksPerGrid = threads / threadsPerBlock;

    assert(blocksPerGrid * threadsPerBlock == threads);

    //std::cout << "blocks per grid: " << blocksPerGrid << std::endl;
    unsigned* solution_counter_mem = NULL;

    gpuErrchk(cudaMalloc((void**)&solution_counter_mem, sizeof(unsigned) * blocksPerGrid));
    gpuErrchk(cudaMemset(solution_counter_mem, 0, sizeof(unsigned) * blocksPerGrid));
  
    uint64_t clause_size = sizeof(GPUClause) * clauses.size();
    assert(clause_size <= CU_CONST_MEMORY_SIZE);

    gpuErrchk(cudaMemcpyToSymbol(clause_data, &clauses[0], clause_size));

    solveComponent<<<blocksPerGrid, threadsPerBlock>>>(solution_counter_mem, clauses.size(), scale_factor);

    gpuErrchk(cudaDeviceSynchronize());

    unsigned* solution_counter_result = (unsigned*)malloc(sizeof(unsigned) * blocksPerGrid);
    assert(solution_counter_result != NULL);

    gpuErrchk(cudaMemcpy(solution_counter_result, solution_counter_mem, sizeof(unsigned) * blocksPerGrid, cudaMemcpyDeviceToHost));

    unsigned long long solutions = 0;
    for (int64_t i = 0; i < blocksPerGrid; i++) {
        solutions += solution_counter_result[i];
    }

    cudaFree(solution_counter_mem);
    free(solution_counter_result);
    return solutions;
}
