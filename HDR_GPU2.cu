/*
 Author: Hamed Hassani Saadi
 April 2017
 Email: hassanih@mcmaster.ca
 to run on monk/mon54:
 module load opencv/2.4.9
 nvcc HDR_GPU2.cu -o HDR_GPU2.o -L /opt/sharcnet/opencv/2.4.9/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64 -I /opt/sharcnet/cuda/7.5.18/include/ -I /opt/sharcnet/opencv/2.4.9/include
 ./HDR_CPU2.o input_file.png
 
 for profililng:
 nvprof ./HDR_GPU2.o input_file.png
 or
 nvprof --print-gpu-trace ./HDR_GPU2.o input_file.png
 or
 nvprof --print-gpu-trace --metrics achieved_occupancy  ./HDR_GPU2.o input_file2.png
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>
#include <algorithm>
#include <cassert>
#include <string.h>
#include <iostream>
#include <thrust/extrema.h>
#include <sys/time.h>
#include <time.h>
#include "utils.h"

#define NUMBINS 1024
#define BLOCKSIZE 1024
/*
 +++++++++++++++++++++timevalu_subtract Function+++++++++++++++++++++
 */
int timeval_subtract (double *result, struct timeval *x, struct timeval *y) {
    struct timeval result0;
    /* Perform the carry for the later subtraction by updating y. */
    if (x->tv_usec < y->tv_usec) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }
    if (x->tv_usec - y->tv_usec > 1000000) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000;
        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }
    /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
    result0.tv_sec = x->tv_sec - y->tv_sec;
    result0.tv_usec = x->tv_usec - y->tv_usec;
    *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;
    /* Return 1 if result is negative. */
    return x->tv_sec < y->tv_sec;
}
/*
 +++++++++++++++++++++LoadImage Function+++++++++++++++++++++
 */
void loadImageHDR(const std::string &filename, float **imagePtr, size_t *numRows, size_t *numCols){
    cv::Mat originImg = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
    cv::Mat image;
    if(originImg.type() != CV_32FC3){
        originImg.convertTo(image,CV_32FC3);
    } else{
        image = originImg;
    }
    if (image.empty()){
        std::cerr << "Couldn't open file: " << filename << std::endl;
        exit(1);
    }
    if (image.channels() != 3){
        std::cerr << "Image must be color!" << std::endl;
        exit(1);
    }
    if (!image.isContinuous()){
        std::cerr << "Image isn't continuous!" << std::endl;
        exit(1);
    }
    *imagePtr = new float[image.rows * image.cols * image.channels()];
    float *cvPtr = image.ptr<float>(0);
    for (int i = 0; i < image.rows * image.cols * image.channels(); ++i)
        (*imagePtr)[i] = cvPtr[i];
    *numRows = image.rows;
    *numCols = image.cols;
}
/*
 +++++++++++++++++++++saveImage Function+++++++++++++++++++++
 */
void saveImageHDR(const float* const image, const size_t numRows, const size_t numCols, const std::string &output_file){
    int sizes[2];
    sizes[0] = (int)numRows;
    sizes[1] = (int)numCols;
    cv::Mat imageHDR(2, sizes, CV_32FC3, (void *)image);
    imageHDR = imageHDR * 255;
    cv::imwrite(output_file.c_str(), imageHDR);
}
/*
 +++++++++++++++++++++compareImages Function+++++++++++++++++++++
 */
void compareImages(std::string reference_filename, std::string test_filename)
{
  cv::Mat reference = cv::imread(reference_filename, -1);
  cv::Mat test = cv::imread(test_filename, -1);

  cv::Mat diff = abs(reference - test);

  cv::Mat diffSingleChannel = diff.reshape(1, 0); //convert to 1 channel, same # rows

  double minVal, maxVal;

  cv::minMaxLoc(diffSingleChannel, &minVal, &maxVal, NULL, NULL); //NULL because we don't care about location

  //now perform transform so that we bump values to the full range

//  diffSingleChannel = (diffSingleChannel - minVal) * (255. / (maxVal - minVal));

//  diff = diffSingleChannel.reshape(reference.channels(), 0);

//  cv::imwrite("differenceImage.png", diff);
  //OK, now we can start comparing values...
  unsigned char *referencePtr = reference.ptr<unsigned char>(0);
  unsigned char *testPtr = test.ptr<unsigned char>(0);

  //checkResultsEps(referencePtr, testPtr, reference.rows * reference.cols * reference.channels(), 4.0, 4.0);
  //checkResultsAutodesk(referencePtr, testPtr, reference.rows * reference.cols * reference.channels(), 0.0, 0);
  checkResultsExact(referencePtr, testPtr, reference.rows * reference.cols * reference.channels(), 50);

  std::cout << "Images are equal 100%." << std::endl;
  return;
}
/*
 +++++++++++++++++++++rgb_to_xyY Function+++++++++++++++++++++
 */
void rgb_to_xyY(float* red, float* green, float* blue, float* x_, float* y_, float* log_Y_, float  delta, size_t numRows, size_t numCols){
    float r, g, b;
    float X,Y,Z, L;
    for (size_t i=0; i<numRows; i++) {
        for (size_t j=0; j<numCols; j++){
            r = red[numCols*i+j];
            g = green[numCols*i+j];
            b = blue[numCols*i+j];
            X = ( r * 0.4124f ) + ( g * 0.3576f ) + ( b * 0.1805f );
            Y = ( r * 0.2126f ) + ( g * 0.7152f ) + ( b * 0.0722f );
            Z = ( r * 0.0193f ) + ( g * 0.1192f ) + ( b * 0.9505f );
            L = X + Y + Z;
            (x_)[numCols*i+j]     = X / L;
            (y_)[numCols*i+j]     = Y / L;
            (log_Y_)[numCols*i+j] = log10f( delta + Y );
        }
    }
}
/*
 +++++++++++++++++++GPU rgb_to_xyY Function+++++++++++++++++++
 */
__global__ void rgb_to_xyY_gpu(float* d_r, float* d_g, float* d_b, float* d_x, float* d_y, float* d_log_Y, float  delta, int num_pixels_y, int num_pixels_x){
    int  ny             = num_pixels_y;
    int  nx             = num_pixels_x;
    int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
    int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;
    if ( image_index_2d.x < nx && image_index_2d.y < ny ){
        float r = d_r[ image_index_1d ];
        float g = d_g[ image_index_1d ];
        float b = d_b[ image_index_1d ];
        
        float X = ( r * 0.4124f ) + ( g * 0.3576f ) + ( b * 0.1805f );
        float Y = ( r * 0.2126f ) + ( g * 0.7152f ) + ( b * 0.0722f );
        float Z = ( r * 0.0193f ) + ( g * 0.1192f ) + ( b * 0.9505f );
        
        float L = X + Y + Z;
        float x = X / L;
        float y = Y / L;
        
        float log_Y = log10f( delta + Y );
        
        d_x[ image_index_1d ]     = x;
        d_y[ image_index_1d ]     = y;
        d_log_Y[ image_index_1d ] = log_Y;
    }
}
/*
 +++++++++++++++++++++histogram_and_prefixsum Function+++++++++++++++++++++
 */
void histogram_and_prefixsum(float *luminance, unsigned int *cdf, size_t numRows, size_t numCols, size_t numBins, float *luminance_min, float *luminance_max){
    float logLumMin = luminance[0];
    float logLumMax = luminance[0];
    //Step 1
    //first we find the minimum and maximum across the entire image
    for (size_t i = 1; i < numCols * numRows; ++i) {
        logLumMin = std::min(luminance[i], logLumMin);
        logLumMax = std::max(luminance[i], logLumMax);
    }
    //Step 2
    float logLumRange = logLumMax - logLumMin;
    *luminance_min = logLumMin;
    *luminance_max = logLumMax;
    //Step 3
    //next we use the now known range to compute
    //a histogram of numBins bins
    unsigned int *histo = new unsigned int[numBins];
    for (size_t i = 0; i < numBins; ++i) histo[i] = 0;
    for (size_t i = 0; i < numCols * numRows; ++i) {
        unsigned int bin = std::min(static_cast<unsigned int>(numBins - 1),
                                    static_cast<unsigned int>((luminance[i] - logLumMin) / logLumRange * numBins));
        histo[bin]++;
    }
    //Step 4
    //finally we perform and exclusive scan (prefix sum)
    //on the histogram to get the cumulative distribution
    (cdf)[0] = 0;
    for (size_t i = 1; i < numBins; ++i) {
        (cdf)[i] = (cdf)[i - 1] + histo[i - 1];
    }
    delete[] histo;
}
/*
 +++++++++++++++++++GPU histogram_and_prefixsum Function+++++++++++++++++++
 */
__global__ void shmem_reduce_kernel_minmax(float * d_out_min, float * d_out_max, float * d_in_min, float * d_in_max, int nblocks){
    extern __shared__ float sdata[];
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;
    if (tid < nblocks){
        sdata[tid] = d_in_min[myId];
        sdata[tid+blockDim.x] = d_in_max[myId];
    }
    else {
        sdata[tid] = 1E+37;
        sdata[tid+blockDim.x] = 1E-37;
    }
    __syncthreads();            // make sure entire block is loaded!
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2){
        if (tid < s){
            sdata[tid] = min(sdata[tid],sdata[tid+s]);
            sdata[tid+blockDim.x] = max(sdata[tid+blockDim.x],sdata[tid+s+blockDim.x]);
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }
    if (tid == 0){
        d_out_min[blockIdx.x] = sdata[0];
        d_out_max[blockIdx.x] = sdata[0+blockDim.x];
    }
}
void reduce_min_max(float * d_log_min, float * d_log_max, float* d_in, float * d_intermin, float * d_intermax, size_t numRows, size_t numCols){
    const int maxThreadsPerBlock = BLOCKSIZE;
    int threads = maxThreadsPerBlock; // launch one thread for each block in prev step
    int blocks = numRows*numCols / maxThreadsPerBlock;
    shmem_reduce_kernel_minmax<<<blocks, threads, 2 * threads * sizeof(float)>>>(d_intermin, d_intermax, d_in, d_in, threads);
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    shmem_reduce_kernel_minmax<<<blocks, 128, 2 * 128 * sizeof(float)>>>(d_log_min, d_log_max, d_intermin, d_intermax, threads);
}
__global__ void simple_histo(unsigned int *d_bins, float * d_log_min, float * d_log_max, float* d_in, size_t numBins, size_t numRows, size_t numCols)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if (myId >= numRows*numCols)
        return;
    if (myId < numBins)
        d_bins[myId] = 0;
    __syncthreads();
    float myBinF = (d_in[myId] - d_log_min[0]) / (d_log_max[0] - d_log_min[0]);
    int myBin =  myBinF * numBins;
    atomicAdd(&(d_bins[myBin]), 1);
}
__global__ void scan(unsigned int *g_odata, unsigned int *g_idata, int n){
    extern __shared__ unsigned int temp[];  // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;
    temp[2*thid] = g_idata[2*thid]; // load input into shared memory
    temp[2*thid+1] = g_idata[2*thid+1];
    for (int d = n>>1; d > 0; d >>= 1){                    // build sum in place up the tree
        __syncthreads();
        if (thid < d){
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (thid == 0) { temp[n - 1] = 0; } // clear the last element
    for (int d = 1; d < n; d *= 2){ // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (thid < d){
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            unsigned int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    g_odata[2*thid] = temp[2*thid]; // write results to device memory
    g_odata[2*thid+1] = temp[2*thid+1];
}
void histogram_and_prefixsum_gpu(float* d_logLuminance, unsigned int * d_cdf, float * d_log_min, float * d_log_max, float * d_intermin, float * d_intermax, unsigned int * d_hist, size_t numRows, size_t numCols, size_t numBins){
    
    reduce_min_max(d_log_min, d_log_max, d_logLuminance, d_intermin, d_intermax, numRows, numCols);
    cudaDeviceSynchronize();
    
    simple_histo<<<numRows*numCols/BLOCKSIZE,BLOCKSIZE>>>(d_hist, d_log_min, d_log_max, d_logLuminance, numBins, numRows, numCols);
    cudaDeviceSynchronize();
    
    scan<<<1,numBins/2,numBins*sizeof(unsigned int)>>>(d_cdf,d_hist,numBins);
    cudaDeviceSynchronize();
}

/*
 +++++++++++++++++++++normalize_cdf Function+++++++++++++++++++++
 */
void normalize_cdf(unsigned int* input_cdf, float* output_cdf, size_t n){
    const float normalization_constant = 1.f / input_cdf[n - 1];
    float tmp;
    for (size_t i=0; i<n; i++) {
        tmp = input_cdf[i]*normalization_constant;
        (output_cdf)[i] = tmp;
    }
}
/*
 +++++++++++++++++++GPU normalize_cdf Function+++++++++++++++++++
 */
__global__ void normalize_cdf_gpu(unsigned int* d_input_cdf, float* d_output_cdf, int n){
    const float normalization_constant = 1.f / d_input_cdf[n - 1];
    int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    if ( global_index_1d < n )
    {
        unsigned int input_value  = d_input_cdf[ global_index_1d ];
        float        output_value = input_value * normalization_constant;
        d_output_cdf[ global_index_1d ] = output_value;
    }
}
/*
 +++++++++++++++++++++tonemap Function+++++++++++++++++++++
 */
void tonemap(float* x, float* y, float* log_Y, float* ncdf, float* r_new, float* g_new, float* b_new, float min_log_Y, float max_log_Y, size_t num_bins, size_t numRows, size_t numCols){
    float log_Y_range = max_log_Y - min_log_Y;
    float x_, y_, log_Y_;
    unsigned int bin_index;
    float X_new, Y_new, Z_new;
    for (size_t i=0; i<numRows; i++) {
        for (size_t j=0; j<numCols; j++) {
            x_ = x[numCols*i+j];
            y_ = y[numCols*i+j];
            log_Y_ = log_Y[numCols*i+j];
            
            bin_index = min((int)num_bins - 1, int((num_bins * (log_Y_ - min_log_Y)) / log_Y_range));
            
            Y_new = ncdf[bin_index];
            X_new = x_ * ( Y_new / y_ );
            Z_new = (1 - x_ - y_) * (Y_new / y_);
            
            (r_new)[numCols*i+j] = ( X_new *  3.2406f ) + ( Y_new * -1.5372f ) + ( Z_new * -0.4986f );
            (g_new)[numCols*i+j] = ( X_new * -0.9689f ) + ( Y_new *  1.8758f ) + ( Z_new *  0.0415f );
            (b_new)[numCols*i+j] = ( X_new *  0.0557f ) + ( Y_new * -0.2040f ) + ( Z_new *  1.0570f );
        }
    }
}
/*
 +++++++++++++++++++GPU tonemap Function+++++++++++++++++++
 */
__global__ void tonemap_gpu(float* d_x, float* d_y, float* d_log_Y, float* d_cdf_norm, float* d_r_new, float* d_g_new, float* d_b_new, float* d_log_min, float* d_log_max, int num_bins, int num_pixels_y, int num_pixels_x){
    float log_Y_range = d_log_max[0] - d_log_min[0];
    int  ny             = num_pixels_y;
    int  nx             = num_pixels_x;
    int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
    int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;
    if ( image_index_2d.x < nx && image_index_2d.y < ny )
    {
        float x         = d_x[ image_index_1d ];
        float y         = d_y[ image_index_1d ];
        float log_Y     = d_log_Y[ image_index_1d ];
        int   bin_index = min( num_bins - 1, int( (num_bins * ( log_Y - d_log_min[0] ) ) / log_Y_range ) );
        float Y_new     = d_cdf_norm[ bin_index ];
        
        float X_new = x * ( Y_new / y );
        float Z_new = ( 1 - x - y ) * ( Y_new / y );
        
        float r_new = ( X_new *  3.2406f ) + ( Y_new * -1.5372f ) + ( Z_new * -0.4986f );
        float g_new = ( X_new * -0.9689f ) + ( Y_new *  1.8758f ) + ( Z_new *  0.0415f );
        float b_new = ( X_new *  0.0557f ) + ( Y_new * -0.2040f ) + ( Z_new *  1.0570f );
        
        d_r_new[ image_index_1d ] = r_new;
        d_g_new[ image_index_1d ] = g_new;
        d_b_new[ image_index_1d ] = b_new;
    }
}
/*
 +++++++++++++++++++++CPU_histeq Function+++++++++++++++++++++
 */
void CPU_histeq(float * red, float * green, float * blue, float * r_new, float * g_new, float * b_new, float * x, float * y, float * luminance, unsigned int * cdf, float * ncdf, size_t numRows, size_t numCols, size_t numBins){
    ////////////////////////////////converting RGB to xyY
    rgb_to_xyY(red, green, blue, x, y, luminance, 0.0001f, numRows, numCols);
    //calculating histogram and CDF
    float luminance_min, luminance_max;
    histogram_and_prefixsum(luminance, cdf, numRows, numCols, numBins, &luminance_min, &luminance_max);
    //normalizing CDF
    normalize_cdf(cdf, ncdf, numBins);
    //tone-mapping
    tonemap(x, y, luminance, ncdf, r_new, g_new, b_new, luminance_min, luminance_max, numBins, numRows, numCols);
}
/*
 +++++++++++++++++++++GPU_histeq Function+++++++++++++++++++++
 */
void GPU_histeq(float * d_red, float * d_green, float * d_blue, float * d_r_new, float * d_g_new, float * d_b_new, float * d_x, float * d_y, float * d_luminance, unsigned int * d_cdf, float * d_ncdf, float * d_log_min, float * d_log_max, float * d_intermin, float * d_intermax, unsigned int * d_hist, size_t numRows, size_t numCols, size_t numBins){
    //convert from RGB space to chrominance/luminance space xyY
    dim3 blockSize(32, 16, 1);
    dim3 gridSize( (numCols + blockSize.x - 1) / blockSize.x, (numRows + blockSize.y - 1) / blockSize.y, 1);
    rgb_to_xyY_gpu<<<gridSize, blockSize>>>(d_red, d_green, d_blue, d_x, d_y, d_luminance, 0.0001f, numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    histogram_and_prefixsum_gpu(d_luminance, d_cdf, d_log_min, d_log_max, d_intermin, d_intermax, d_hist, numRows, numCols, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
    normalize_cdf_gpu<<<(numBins + 192 - 1) / 192, 192>>>(d_cdf, d_ncdf, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    dim3 blockSize1(32, 16, 1);
    dim3 gridSize1( (numCols + blockSize.x - 1) / blockSize.x, (numRows + blockSize.y - 1) / blockSize.y );
    tonemap_gpu<<<gridSize1, blockSize1>>>(d_x, d_y, d_luminance, d_ncdf, d_r_new, d_g_new, d_b_new, d_log_min, d_log_max, numBins, numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
/*
 +++++++++++++++++++++++++++++++++++++++++++++++++++++++
 +++++++++++++++++++++Main Function+++++++++++++++++++++
 +++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */
int main(int argc, const char * argv[]) {
   	std::string input_file;
    std::string output_file_cpu;
    std::string output_file_gpu;
	if(argc == 2){
    	input_file = std::string(argv[1]);
    }
	else{
    	input_file = "input_file.png";
	}
    std::size_t idx = input_file.find_last_of("/");
    if (idx == std::string::npos) {
        output_file_cpu = "cpu_" + input_file;
        output_file_gpu = "gpu_" + input_file;
    }
    else{
        output_file_cpu = "cpu_" + input_file.substr(idx+1,input_file.size()-idx);
        output_file_gpu = "gpu_" + input_file.substr(idx+1,input_file.size()-idx);
        
    }
    
    struct timeval td0, td1;
    double t_cpu, t_gpu;
    double t_copyin, t_copyout;
    
    ////////////////////////////////loading iamge
    float *imgPtr;
    size_t numRows, numCols;
    loadImageHDR(input_file, &imgPtr, &numRows, &numCols);
    ////////////////////////////////separating RGB channels
    size_t numPixels = numRows * numCols;
    float *red   = new float[numPixels];
    float *green = new float[numPixels];
    float *blue  = new float[numPixels];
    for (size_t i = 0; i < numPixels; ++i) {
        blue[i]  = imgPtr[3 * i + 0];
        green[i] = imgPtr[3 * i + 1];
        red[i]   = imgPtr[3 * i + 2];
    }
    delete[] imgPtr;

    /*
     //////////////////////
     ///////////////////CPU
     //////////////////////
     */
    //image histogram equalization on CPU
    float * r_new = new float[numPixels];
    float * g_new = new float[numPixels];
    float * b_new = new float[numPixels];
    float * x = new float[numPixels];
    float * y = new float[numPixels];
    float * luminance = new float[numPixels];
    size_t numBins = NUMBINS;
    unsigned int * cdf = new unsigned int[numBins];
    float * ncdf = new float[numBins];

    gettimeofday (&td0, NULL);
    CPU_histeq(red, green, blue, r_new, g_new, b_new, x, y, luminance, cdf, ncdf, numRows, numCols, numBins);
    gettimeofday (&td1, NULL);
    timeval_subtract (&t_cpu, &td1, &td0);
 
    delete[] x;
    delete[] y;
    delete[] luminance;
    delete[] cdf;
    delete[] ncdf;
    /*
     //////////////////////
     ///////////////////GPU
     //////////////////////
     */
    //make sure the context initializes ok
    checkCudaErrors(cudaFree(0));
    
    float * d_red, * d_green, * d_blue;
    float * d_x, * d_y, * d_luminance;
    unsigned int * d_cdf;
    float * d_log_min, * d_log_max;
    float * d_ncdf;
    float * d_r_new, * d_g_new, * d_b_new;
    float * d_intermin, * d_intermax;
    unsigned int * d_hist;
    checkCudaErrors(cudaMalloc(&d_red, sizeof(float)*numPixels));
    checkCudaErrors(cudaMalloc(&d_green, sizeof(float)*numPixels));
    checkCudaErrors(cudaMalloc(&d_blue, sizeof(float)*numPixels));
    checkCudaErrors(cudaMalloc(&d_x, sizeof(float)*numPixels));
    checkCudaErrors(cudaMalloc(&d_y, sizeof(float)*numPixels));
    checkCudaErrors(cudaMalloc(&d_luminance, sizeof(float)*numPixels));
    checkCudaErrors(cudaMalloc(&d_log_min, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_log_max, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_cdf, sizeof(unsigned int)*numBins));
    checkCudaErrors(cudaMalloc(&d_ncdf, sizeof(float)*numBins));
    checkCudaErrors(cudaMalloc(&d_r_new, sizeof(float)*numPixels));
    checkCudaErrors(cudaMalloc(&d_g_new, sizeof(float)*numPixels));
    checkCudaErrors(cudaMalloc(&d_b_new, sizeof(float)*numPixels));
    checkCudaErrors(cudaMalloc(&d_intermin, sizeof(float)*numRows*numCols));
    checkCudaErrors(cudaMalloc(&d_intermax, sizeof(float)*numRows*numCols));
    checkCudaErrors(cudaMalloc(&d_hist, sizeof(unsigned int)*numBins));

    gettimeofday (&td0, NULL);
    checkCudaErrors(cudaMemcpy(d_red, red, sizeof(float)*numPixels, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_green, green, sizeof(float)*numPixels, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_blue, blue, sizeof(float)*numPixels, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    gettimeofday (&td1, NULL);
    timeval_subtract (&t_copyin, &td1, &td0);

    gettimeofday (&td0, NULL);
    GPU_histeq(d_red, d_green, d_blue, d_r_new, d_g_new, d_b_new, d_x, d_y, d_luminance, d_cdf, d_ncdf, d_log_min, d_log_max, d_intermin, d_intermax, d_hist, numRows, numCols, numBins);
    gettimeofday (&td1, NULL);
    timeval_subtract (&t_gpu, &td1, &td0);
    
    float * h_red = new float[numPixels];
    float * h_green = new float[numPixels];
    float * h_blue = new float[numPixels];

    gettimeofday (&td0, NULL);
    checkCudaErrors(cudaMemcpy(h_red, d_r_new, sizeof(float) * numPixels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_green, d_g_new, sizeof(float) * numPixels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_blue, d_b_new, sizeof(float) * numPixels, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    gettimeofday (&td1, NULL);
    timeval_subtract (&t_copyout, &td1, &td0);
    
    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_green));
    checkCudaErrors(cudaFree(d_blue));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_luminance));
    checkCudaErrors(cudaFree(d_log_min));
    checkCudaErrors(cudaFree(d_log_max));
    checkCudaErrors(cudaFree(d_cdf));
    checkCudaErrors(cudaFree(d_ncdf));
    checkCudaErrors(cudaFree(d_r_new));
    checkCudaErrors(cudaFree(d_g_new));
    checkCudaErrors(cudaFree(d_b_new));
    checkCudaErrors(cudaFree(d_intermin));
    checkCudaErrors(cudaFree(d_intermax));
    checkCudaErrors(cudaFree(d_hist));

    //recombine the image channels
    float *imageHDR = new float[numPixels * 3];
    for (size_t i = 0; i < numPixels; ++i) {
        imageHDR[3 * i + 0] = b_new[i];
        imageHDR[3 * i + 1] = g_new[i];
        imageHDR[3 * i + 2] = r_new[i];
    }
    //saving image
    saveImageHDR(imageHDR, numRows, numCols, output_file_cpu);

    for (size_t i = 0; i < numPixels; ++i) {
        imageHDR[3 * i + 0] = h_blue[i];
        imageHDR[3 * i + 1] = h_green[i];
        imageHDR[3 * i + 2] = h_red[i];
    }
    //saving image
	saveImageHDR(imageHDR, numRows, numCols, output_file_gpu);

    printf("CPU runtime: %f ms\n",t_cpu);
    printf("GPU runtime: %f ms\n",t_gpu);
    printf("Copying data into the GPU: %f ms\n", t_copyin);
    printf("Copying data form the GPU: %f ms\n", t_copyout);
    printf("GPU runtime + data transfer: %f ms\n", t_copyin+t_gpu+t_copyout);
    printf("Image dimension: %dx%d = %d pixels\n", (int)numRows, (int)numCols, (int)numPixels);
    compareImages(output_file_cpu, output_file_gpu);
    
    delete[] red;
    delete[] green;
    delete[] blue;
    delete[] r_new;
    delete[] g_new;
    delete[] b_new;
	delete[] imageHDR;
    delete[] h_red;
    delete[] h_green;
    delete[] h_blue;
}