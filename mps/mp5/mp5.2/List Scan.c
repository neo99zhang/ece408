// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, int step) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int bd=blockDim.x;
  int startX;
  int spaceX;
  int newStartX=bd * bx * 2 + tx;
  
  __shared__ float T[2 * BLOCK_SIZE];
  
  if (step){//if step 1
    startX = 2 * bd * (tx + 1) - 1;
    spaceX = bd * 2;
  }
  else{//if step 0
    startX = 2 * bx * bd + tx;
    spaceX = bd;    
  }
  
  //load two piece
  for (int i = 0; i < 2; i++){
    T[tx + bd * i] = ((spaceX * i + startX) < len) ? input[spaceX * i + startX]:0;
  }
  
  //first scan
  int stride = 1;
  while(stride < 2 * BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index < 2 * BLOCK_SIZE && (index - stride) >= 0){
      T[index] += T[index - stride];
    }
    stride = stride * 2;
  }
  
  //second scan
  stride = BLOCK_SIZE / 2;
  while(stride > 0) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 -1;
    if ((index+stride) < 2 * BLOCK_SIZE)
      T[index + stride] += T[index];
    stride = stride / 2;
  }

  __syncthreads();
  
  //copy the inBlock sum to output
  for (int i = 0; i < 2; i++){
    output[bd * i + newStartX] = ((bd * i + newStartX) < len) ? T[tx + bd * i] : 0;
  }
  
}

__global__ void add(float *input, float *output, float *sum, int len){
  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int bd=blockDim.x;
  int newStartX=bd * bx * 2 + tx;
  
  __shared__ float shift;
  
  if (tx == 0){
    if (bx == 0){
      shift = 0;
    }
    else{
      shift = sum[bx - 1];
    }
  }
  
  __syncthreads();
  
  for (int i = 0; i < 2; i++){                
    output[newStartX + bd * i] = input[newStartX + bd * i] + shift;
  }
}
int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *sumBlocks;
  float *deviceBuffer;
  int numElements; // number of elements in the list
  int numBlocks;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  numBlocks=ceil(numElements/(2.0*BLOCK_SIZE));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceBuffer, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&sumBlocks, 2 * BLOCK_SIZE * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(numBlocks,1,1);
  dim3 dimBlock(BLOCK_SIZE,1,1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid,dimBlock>>>
    (deviceInput,deviceBuffer,numElements,0);
  
  cudaDeviceSynchronize();
  
  dim3 dimGrid2(1,1,1);
  scan<<<dimGrid2,dimBlock>>>
    (deviceBuffer,sumBlocks,numElements,1);
  
  cudaDeviceSynchronize();
  
  add<<<dimGrid,dimBlock>>>
    (deviceBuffer,deviceOutput,sumBlocks,numElements);
  
  cudaDeviceSynchronize();
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceBuffer);
  cudaFree(sumBlocks);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
