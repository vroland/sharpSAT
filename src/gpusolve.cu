#include <cuda_runtime.h>
#include <cuda.h>
#include "gputypes.h"
#include <signal.h>
#include <vector>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

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

__device__ uint32_t get_global_id() {
    // TODO: y and z
    return blockDim.x * blockIdx.x + threadIdx.x;
}

__global__ void solveComponent(unsigned* solution_counter, GPUClause* clauses, unsigned clause_count, uint64_t threads) {
    uint32_t id = get_global_id();
    if (id >= threads) {
        return;
    }
    for (unsigned clause_idx = 0; clause_idx < clause_count; clause_idx++) {
        GPUClause clause = clauses[clause_idx];
        // flip negatively signed variables
        uint32_t signed_assignment = id ^ clause.neg_vars;
        // clause is not satisfied -> abort
        if ((signed_assignment & clause.vars) == 0) return;
    }
    atomicAdd(solution_counter, 1);
}

unsigned componentModelCount(const std::vector<GPUClause>& clauses, unsigned variable_count) {
    int64_t threadsPerBlock = 512;
    int64_t threads = 1 << variable_count;
    int64_t blocksPerGrid = (threads + threadsPerBlock - 1) / threadsPerBlock;

    GPUClause* clauses_mem = NULL;
    unsigned* solution_counter_mem = NULL;

    gpuErrchk(cudaMalloc((void**)&solution_counter_mem, sizeof(unsigned)));
    
    gpuErrchk(cudaMalloc((void**)&clauses_mem, sizeof(GPUClause) * clauses.size()));
    gpuErrchk(cudaMemcpy(clauses_mem, &clauses[0], sizeof(GPUClause) * clauses.size(), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(solution_counter_mem, 0, sizeof(unsigned)));

    solveComponent<<<blocksPerGrid, threadsPerBlock>>>(solution_counter_mem, clauses_mem, clauses.size(), threads);

    gpuErrchk(cudaDeviceSynchronize());

    unsigned solution_counter_result = 0;
    gpuErrchk(cudaMemcpy(&solution_counter_result, solution_counter_mem, sizeof(unsigned), cudaMemcpyDeviceToHost));

    cudaFree(clauses_mem);
    cudaFree(solution_counter_mem);
    return solution_counter_result;
}
