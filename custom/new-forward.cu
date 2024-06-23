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


#define UNROLL_TILE_WIDTH 16
__global__ void unroll_conv_forward_kernel(float *y, const float *x, const float* k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    __shared__ float subTileM[UNROLL_TILE_WIDTH][UNROLL_TILE_WIDTH];
    __shared__ float subTileN[UNROLL_TILE_WIDTH][UNROLL_TILE_WIDTH];


    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b = blockIdx.z;
    int W_unroll = H_out * W_out;
    int H_unroll = C * K * K;
    int numAColumns = H_unroll;
    int m_upper = (numAColumns + UNROLL_TILE_WIDTH - 1) / UNROLL_TILE_WIDTH;

    int row = by * UNROLL_TILE_WIDTH + ty;
    int col = bx * UNROLL_TILE_WIDTH + tx;

    float result = 0;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    for (int m = 0; m < m_upper; m++) {
        // Fill the shared memory
        int m_col = (m * UNROLL_TILE_WIDTH + tx);
        if (m_col < numAColumns && row < M) {
            subTileM[ty][tx] = k4d(row, m_col/(K*K), (m_col%(K*K))/K, (m_col%(K*K))%K);
        }
        else {
            subTileM[ty][tx] = 0;
        }
        int n_row = (m * UNROLL_TILE_WIDTH + ty);
        int x_c = n_row / (K * K);
        int x_base_h = col / W_out;
        int x_base_w = col % W_out;
        int x_p = (n_row % (K * K)) / K;
        int x_q = (n_row % (K * K)) % K;
        if (n_row < numAColumns && col < W_unroll) {
            subTileN[ty][tx] = x4d(b, x_c, x_base_h + x_p, x_base_w + x_q);
        }
        else {
            subTileN[ty][tx] = 0;
        }
        __syncthreads();
        for (int k = 0; k < UNROLL_TILE_WIDTH; k++)
        {
            result += subTileM[ty][k] * subTileN[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < W_unroll) {
        y4d(b, row, col / W_out, col % W_out) = result;
    }

#undef y4d
#undef x4d
#undef k4d
}

__constant__ float mask[4096]; // >= 16*4*7*7
#define SHARE_TILE_WIDTH 16
// We make sure that each kernel handle TILE_WIDTH*TILE_WIDTH*c output
// From mp4
__global__ void share_conv_forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    extern __shared__ float SM[];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int SM_WIDTH = SHARE_TILE_WIDTH + K - 1;
    const int grid_subh = ceil((float)H_out / (float)SHARE_TILE_WIDTH);
    const int grid_subw = ceil((float)W_out / (float)SHARE_TILE_WIDTH);
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int z = blockIdx.z;
    const int tile_h = z / grid_subw;
    const int tile_w = z % grid_subw;
    const int base_h = tile_h * SHARE_TILE_WIDTH;
    const int base_w = tile_w * SHARE_TILE_WIDTH;
    const int h = base_h + threadIdx.y;
    const int w = base_w + threadIdx.x;
    float result = 0;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    for(int c = 0; c < C; c++) {
        // First load our share memory
        for (int iter_h = h; iter_h < base_h + SM_WIDTH; iter_h += SHARE_TILE_WIDTH) {
            for (int iter_w = w; iter_w < base_w + SM_WIDTH; iter_w += SHARE_TILE_WIDTH) {
                const int sm_h = iter_h - base_h;
                const int sm_w = iter_w - base_w;
                const int sm_idx = sm_h * SM_WIDTH + sm_w;
                if (b < B && iter_h < H && iter_w < W) {
                    SM[sm_idx] = x4d(b, c, iter_h, iter_w);
                }
                else {
                    SM[sm_idx] = 0.0f;
                }
            }
        }
            
        __syncthreads();
        // Compute!
        #pragma unroll
        for (int p = 0; p < K; p++) {
            #pragma unroll
            for (int q = 0; q < K; q++) {
                result += SM[(threadIdx.y+p) * SM_WIDTH + (threadIdx.x+q)]  * k4d(m, c, p, q);
            }
        }
        __syncthreads();
    }

    if(b < B && m < M && h < H_out && w < W_out) {
        y4d(b, m, h, w) = result;
    }

#undef y4d
#undef x4d
#undef k4d
}

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Declare relevant device pointers
    float *device_y;
    float *device_x;
    float *device_k;

    // Init size
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int y_size = B * M * H_out * W_out;
    int x_size = B * C * H * W;
    int k_size = M * C * K * K;
    int y_bytes = y_size * sizeof(float);
    int x_bytes = x_size * sizeof(float);
    int k_bytes = k_size * sizeof(float);
    int W_unroll = H_out * W_out;
    int H_unroll = C * K * K;

    printf("B: %d, M: %d, C: %d, H: %d, W: %d, K: %d, H_out: %d, W_out: %d, W_unroll: %d, H_unroll: %d\n", B, M, C, H, W, K, H_out, W_out, W_unroll, H_unroll);

    // Allocate memory and copy over the relevant data structures to the GPU
    cudaCheck(cudaMalloc((void **)&device_y, y_bytes));
    cudaCheck(cudaMalloc((void **)&device_x, x_bytes));
    cudaCheck(cudaMalloc((void **)&device_k, k_bytes));

    cudaCheck(cudaMemcpy((void *)device_x, (void *)host_x, x_bytes, cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpyToSymbol(mask, host_k, k_bytes));
    cudaCheck(cudaMemcpy((void *)device_k, (void *)host_k, k_bytes, cudaMemcpyHostToDevice));

    // std::cout << "Done Allocate memory\n";

    // // Set up unroll matrix
    // float *device_x_unroll;
    // int W_unroll = H_out * W_out;
    // int H_unroll = C * K * K;
    // int x_unroll_bytes = W_unroll * H_unroll * sizeof(float);
    // cudaCheck(cudaMalloc((void **)&device_x_unroll, x_unroll_bytes));

    // dim3 dimUnrollBlock(BLOCK_SIZE);
    // dim3 dimUnrollGrid(ceil((float)(C*H_out*W_out) / (float)BLOCK_SIZE));

    // dim3 dimMulBlock(TILE_WIDTH, TILE_WIDTH);
    // dim3 dimMulGrid(ceil((float)W_unroll / TILE_WIDTH), ceil((float)M / TILE_WIDTH));

    // // Set the kernel dimensions and call the kernel
    // for (int b = 0; b < B; b++) {
    //     unroll_kernel<<<dimUnrollGrid, dimUnrollBlock>>>(device_x, device_x_unroll, b, C, H, W, K, W_unroll, W_out);
    //     matrixMultiplyShared<<<dimMulGrid, dimMulBlock>>>(device_k, device_x_unroll, device_y + b * M * H_out * W_out, H_unroll, M, W_unroll);
    // }

    // std::cout << "Done calling kernel\n";

    if(M == 4) {
        const int grid_subh = ceil((float)H_out / (float)SHARE_TILE_WIDTH);
        const int grid_subw = ceil((float)W_out / (float)SHARE_TILE_WIDTH);
        const int SM_WIDTH = SHARE_TILE_WIDTH + K - 1;
        const int SM_SIZE = SM_WIDTH * SM_WIDTH * sizeof(float);

        dim3 dimBlock(SHARE_TILE_WIDTH, SHARE_TILE_WIDTH);
        dim3 dimGrid(B, M, grid_subh * grid_subw);

        share_conv_forward_kernel<<<dimGrid, dimBlock, SM_SIZE>>>(device_y, device_x, B, M, C, H, W, K);
    } else {
        dim3 dimBlock(UNROLL_TILE_WIDTH, UNROLL_TILE_WIDTH, 1);
        dim3 dimGrid(ceil((float)W_unroll / UNROLL_TILE_WIDTH), ceil((float)M / UNROLL_TILE_WIDTH), B);

        unroll_conv_forward_kernel<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    }

    // Copy the output back to host
    cudaCheck(cudaMemcpy((void *)host_y, (void *)device_y, y_bytes, cudaMemcpyDeviceToHost));

    // std::cout << "Done copy back\n";

    // Free device memory
    cudaCheck(cudaFree(device_y));
    cudaCheck(cudaFree(device_x));
    cudaCheck(cudaFree(device_k));
    // cudaCheck(cudaFree(device_x_unroll));

    // std::cout << "Done free memory\n";

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}