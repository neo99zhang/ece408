#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define cudaCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            printf("[ERROR] Failed to run stmt %s\n", #stmt);                       \
            printf("[ERROR] Got CUDA error %s\n", cudaGetErrorString(err));    \
            return;                                                        \
        }                                                                     \
    } while(0)

#define TILE_WIDTH 32
#define BLOCK_SIZE 32

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numAColumns,
                                     int numCRows, int numCColumns)
{
    int numARows = numCRows;
    int numBRows = numAColumns;
    int numBColumns = numCColumns;
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    int m_upper = (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int m = 0; m < m_upper; m++)
    {
        if ((m * TILE_WIDTH + tx) < numAColumns && row < numARows)
        {
            subTileM[ty][tx] = A[(row)*numAColumns + (TILE_WIDTH * m + tx)];
        }
        else
        {
            subTileM[ty][tx] = 0;
        }
        if (col < numBColumns && (m * TILE_WIDTH + ty) < numBRows)
        {
            subTileN[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + (col)];
        }
        else
        {
            subTileN[ty][tx] = 0;
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
        {
            Pvalue += subTileM[ty][k] * subTileN[k][tx];
        }
        __syncthreads();
    }
    if (row < numCRows && col < numCColumns)
    {
        C[row * numCColumns + col] = Pvalue;
    }
}

__global__ void unroll_kernel(const float *x, float *x_unroll, int b, const int C, const int H, const int W, const int K, const int W_unroll, const int W_out)
{
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t < C * W_unroll) {
        int c = t / W_unroll;
        int s = t % W_unroll;
        int h_out = s / W_out;
        int w_out = s % W_out;
        for (int p = 0; p < K; p++)
            for (int q = 0; q < K; q++) {
                int h_unroll = c * K * K + p * K + q;
                int w_unroll = h_out * W_out + w_out;
                x_unroll[h_unroll * W_unroll + w_unroll] = x4d(b, c, h_out + p, w_out + q);
            }
    }
#undef x4d
}

// __host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//     // Declare relevant device pointers
//     float *device_y;
//     float *device_x;
//     float *device_k;

//     // Init size
//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;
//     int y_size = B * M * H_out * W_out;
//     int x_size = B * C * H * W;
//     int k_size = M * C * K * K;
//     int y_bytes = y_size * sizeof(float);
//     int x_bytes = x_size * sizeof(float);
//     int k_bytes = k_size * sizeof(float);

//     printf("B: %d, M: %d, C: %d, H: %d, W: %d, K: %d, H_out: %d, W_out: %d\n", B, M, C, H, W, K, H_out, W_out);

//     // Allocate memory and copy over the relevant data structures to the GPU
//     cudaCheck(cudaMalloc((void **)&device_y, y_bytes));
//     cudaCheck(cudaMalloc((void **)&device_x, x_bytes));
//     cudaCheck(cudaMalloc((void **)&device_k, k_bytes));

//     cudaCheck(cudaMemcpy((void *)device_x, (void *)host_x, x_bytes, cudaMemcpyHostToDevice));

//     // cudaCheck(cudaMemcpyToSymbol(mask, host_k, k_bytes));
//     cudaCheck(cudaMemcpy((void *)device_k, (void *)host_k, k_bytes, cudaMemcpyHostToDevice));

//     // std::cout << "Done Allocate memory\n";

//     // Set up unroll matrix
//     float *device_x_unroll;
//     int W_unroll = H_out * W_out;
//     int H_unroll = C * K * K;
//     int x_unroll_bytes = W_unroll * H_unroll * sizeof(float);
//     cudaCheck(cudaMalloc((void **)&device_x_unroll, x_unroll_bytes));

//     dim3 dimUnrollBlock(BLOCK_SIZE);
//     dim3 dimUnrollGrid(ceil((float)(C*H_out*W_out) / (float)BLOCK_SIZE));

//     dim3 dimMulBlock(TILE_WIDTH, TILE_WIDTH);
//     dim3 dimMulGrid(ceil((float)W_unroll / TILE_WIDTH), ceil((float)M / TILE_WIDTH));

//     // Set the kernel dimensions and call the kernel
//     for (int b = 0; b < B; b++) {
//         unroll_kernel<<<dimUnrollGrid, dimUnrollBlock>>>(device_x, device_x_unroll, b, C, H, W, K, W_unroll, W_out);
//         matrixMultiplyShared<<<dimMulGrid, dimMulBlock>>>(device_k, device_x_unroll, device_y + b * M * H_out * W_out, H_unroll, M, W_unroll);
//     }

//     // std::cout << "Done calling kernel\n";

//     // Copy the output back to host
//     cudaCheck(cudaMemcpy((void *)host_y, (void *)device_y, y_bytes, cudaMemcpyDeviceToHost));

//     // std::cout << "Done copy back\n";

//     // Free device memory
//     cudaCheck(cudaFree(device_y));
//     cudaCheck(cudaFree(device_x));
//     cudaCheck(cudaFree(device_k));
//     cudaCheck(cudaFree(device_x_unroll));

//     // std::cout << "Done free memory\n";

//     // Useful snippet for error checking
//     cudaError_t error = cudaGetLastError();
//     if(error != cudaSuccess)
//     {
//         std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//         exit(-1);
//     }
// }

// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }