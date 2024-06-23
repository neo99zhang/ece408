// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32

//@@ insert code here
__global__ void float2Uchar(float *input,unsigned char *output,int width,int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if((x < width) && (y < height) ){
        int idx = blockIdx.z * width * height + y * width + x;
        output[idx] = (unsigned char) (HISTOGRAM_LENGTH-1) * input[idx];
    }
    return;
}

__global__ void RGB2Grays(unsigned char *input,unsigned char *output,int width,int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if((x < width) && (y < height) ){
        int idx = y * width + x;
        unsigned char r = input[3 * idx];
        unsigned char g = input[3 * idx + 1];
        unsigned char b = input[3 * idx + 2];
        output[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    }
    return;
}
__global__ void histo_kernel(unsigned char *buffer,
unsigned int *histo,int width, int height)
{
    __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
    int tx = threadIdx.x + threadIdx.y * blockDim.x;
    if (tx< HISTOGRAM_LENGTH) histo_private[tx] = 0;
    __syncthreads();
    int x = threadIdx.x+ blockIdx.x* blockDim.x;
    int y = threadIdx.y+ blockIdx.y* blockDim.y;

    if ( (x < width) && (y < height)) {
        int i = y * width + x;
        atomicAdd( &(histo_private[buffer[i]]), 1);
    }
    __syncthreads();
    if (tx < HISTOGRAM_LENGTH)
    atomicAdd( &(histo[tx]), histo_private[tx] );
}

__global__ void histo2cdf(unsigned int *input, float *output, int width, int height){
   
    int tx = threadIdx.x;

    __shared__ unsigned int T[HISTOGRAM_LENGTH];
    T[tx] = input[tx];

    int stride = 1;
    while(stride < 2 * HISTOGRAM_LENGTH) {
        __syncthreads();
        int index = (tx + 1) * stride * 2 - 1;
        if((index < HISTOGRAM_LENGTH) && ((index - stride) >= 0)){
        T[index] += T[index - stride];
        }
        stride = stride * 2;
    }
    
    //second scan
    stride = HISTOGRAM_LENGTH / 2;
    while(stride > 0) {
        __syncthreads();
        int index = (tx + 1) * stride * 2 -1;
        if ((index+stride) <  HISTOGRAM_LENGTH)
        T[index + stride] += T[index];
        stride = stride / 2;
    }

    __syncthreads();
    output[tx] = T[tx] / ((float) (width * height));
}

__global__ void equalization(unsigned char *input,float *output, float *cdf,int width,int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if((x < width) && (y < height) ){
        int idx = blockIdx.z * width * height + y * width + x;
        unsigned char val = input[idx];
        float temp = 255*(cdf[val] - cdf[0])/(1.0 - cdf[0]);
        float clamp = min(max(temp,0.0),255.0);
        input[idx] = (unsigned char) clamp;
    }

    __syncthreads();

    if((x < width) && (y < height) ){
        int idx = blockIdx.z * width * height + y * width + x;
        output[idx] = (float) (input[idx] / 255.0);
    }
    return;
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  float   *deviceImageFloat;
  unsigned char *deviceImageChar;
  unsigned char *deviceImageGrays;
  unsigned int  *deviceImageHisto;
  float   *deviceImageCdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  hostInputImageData  = wbImage_getData(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void**) &deviceImageFloat,imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceImageChar,imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void**) &deviceImageGrays, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void**) &deviceImageHisto,HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void**) &deviceImageCdf,HISTOGRAM_LENGTH * sizeof(float));

  cudaMemcpy(deviceImageFloat,hostInputImageData,imageHeight*imageWidth*imageChannels*sizeof(float),cudaMemcpyHostToDevice);
  dim3 dimBlock;
  dim3 dimGrid;
  
  dimGrid = dim3(ceil(1.0*imageWidth/BLOCK_SIZE),ceil(1.0*imageHeight/BLOCK_SIZE),imageChannels);
  dimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,1);
  
  float2Uchar<<<dimGrid,dimBlock>>>(deviceImageFloat,deviceImageChar,imageWidth,imageHeight);
  
  cudaDeviceSynchronize();

  dimGrid = dim3(ceil(1.0*imageWidth/BLOCK_SIZE),ceil(1.0*imageHeight/BLOCK_SIZE),1);

  RGB2Grays<<<dimGrid,dimBlock>>>(deviceImageChar,deviceImageGrays,imageWidth,imageHeight);

  cudaDeviceSynchronize();

  histo_kernel<<<dimGrid,dimBlock>>>(deviceImageGrays,deviceImageHisto,imageWidth,imageHeight);

  cudaDeviceSynchronize();

  dimGrid  = dim3(1, 1, 1);
  dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);

  histo2cdf<<<dimGrid,dimBlock>>>(deviceImageHisto,deviceImageCdf,imageWidth,imageHeight);

  cudaDeviceSynchronize();

  dimGrid = dim3(ceil(1.0*imageWidth/BLOCK_SIZE),ceil(1.0*imageHeight/BLOCK_SIZE),imageChannels);
  dimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,1);

  equalization<<<dimGrid,dimBlock>>>(deviceImageChar,deviceImageFloat,deviceImageCdf,imageWidth,imageHeight);

  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData,deviceImageFloat,imageWidth*imageHeight*imageChannels*sizeof(float),cudaMemcpyDeviceToHost);
  
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceImageFloat);
  cudaFree(deviceImageChar);
  cudaFree(deviceImageGrays);
  cudaFree(deviceImageHisto);
  cudaFree(deviceImageCdf);
  return 0;
}
