#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void global_reduce_kernel(float * d_out, float * d_in)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // do reduction in global mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            d_in[myId] += d_in[myId + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
    }
}

// __global__ void shmem_reduce_kernel(float * d_out, const float * d_in)
// {
//     // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
//     extern __shared__ float sdata[];

//     int myId = threadIdx.x + blockDim.x * blockIdx.x;
//     int tid  = threadIdx.x;

//     // load shared mem from global mem
//     sdata[tid] = d_in[myId];
//     __syncthreads();            // make sure entire block is loaded!

//     // do reduction in shared mem
//     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
//     {
//         if (tid < s)
//         {
//             sdata[tid] += sdata[tid + s];
//         }
//         __syncthreads();        // make sure all adds at one stage are done!
//     }

//     // only thread 0 writes result for this block back to global mem
//     if (tid == 0)
//     {
//         d_out[blockIdx.x] = sdata[0];
//     }
// }

// __global__ void optimized_shmem_reduce_kernel(float * d_out, const float * d_in)
// {
//     // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
//     extern __shared__ float sdata[];

//     int myId = threadIdx.x + blockDim.x * blockIdx.x;
//     int tid  = threadIdx.x;

//     // load shared mem from global mem
//     sdata[tid] = d_in[myId];
//     __syncthreads();            // make sure entire block is loaded!


//     // do reduction in shared mem
//     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
//     {
//         if (tid < s)
//         {
//             sdata[tid] += sdata[tid + s];
//         }
//         __syncthreads();        // make sure all adds at one stage are done!
//     }

//     // only thread 0 writes result for this block back to global mem
//     if (tid == 0)
//     {
//         d_out[blockIdx.x] = sdata[0];
//     }
// }

// void reduce(float * d_out, float * d_intermediate, float * d_in,
//             int size, bool usesSharedMemory)
// {
//     // assumes that size is not greater than maxThreadsPerBlock^2
//     // and that size is a multiple of maxThreadsPerBlock
//     const int maxThreadsPerBlock = 1024;
//     int threads = maxThreadsPerBlock;
//     int blocks = size / maxThreadsPerBlock;
//     if (usesSharedMemory)
//     {
//         shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
//             (d_intermediate, d_in);
//     }
//     else
//     {
//         global_reduce_kernel<<<blocks, threads>>>
//             (d_intermediate, d_in);
//     }
//     // now we're down to one block left, so reduce it
//     threads = blocks; // launch one thread for each block in prev step
//     blocks = 1;
//     if (usesSharedMemory)
//     {
//         shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
//             (d_out, d_intermediate);
//     }
//     else
//     {
//         global_reduce_kernel<<<blocks, threads>>>
//             (d_out, d_intermediate);
//     }
// }

// void optimized_reduce(float * d_out, float * d_intermediate, float * d_in, int size)
// {
//     // assumes that size is not greater than maxThreadsPerBlock^2
//     // and that size is a multiple of maxThreadsPerBlock
//     /**
// 	Optimizations mentioned by the course instructor:
// 	- Processing multiple items per thread, instead of just one
// 	- Perform first step of the reduction right when you read the items from global to shared memory
// 	- Take advantage of the fact that warps are synchronous when doing the last steps of the reduction
//     */

//     // const int maxThreadsPerBlock = 1024;
//     const int maxThreadsPerBlock = 4;

//     int threads = maxThreadsPerBlock;
//     int blocks = size / maxThreadsPerBlock;

//     printf("threads = %i\n", threads);
//     printf("blocks = %i\n", blocks);

//     optimized_shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
//        (d_intermediate, d_in);

//     // now we're down to one block left, so reduce it
//     threads = blocks; // launch one thread for each block in prev step
//     blocks = 1;

//     optimized_shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
// 	    (d_out, d_intermediate);
// }

void print_array(const int* arr, const int size)
{
    for (int i = 0; i < size; i++) {
        printf("%i ", arr[i]);
    }
    printf("\n");
}

__global__ void test_kernel(int * d_out, const int * d_in)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ int sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // printf("myId = %i\n", myId);

    // printf("myId = %i, tid = %i\n", myId, tid);

    if (myId == 4) {
        printf("\n");
        printf("threadIdx.x = %i\n", threadIdx.x);
        printf("blockDim.x = %i\n", blockDim.x);
        printf("blockIdx.x = %i\n", blockIdx.x);
        printf("\n");
    }

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // if (myId == 4) {
    //     for (int i = 0; i < blockDim.x; i++) {
    //         printf("%i ", sdata[i]);
    //     }
    //     printf("\n");
    // }

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!

        // if (myId == 4) {
        //     for (int i = 0; i < blockDim.x; i++) {
        //         printf("%i ", sdata[i]);
        //     }
        //     printf("\n");
        // }
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }

    // __syncthreads();        // make sure all adds at one stage are done!

    // if (myId == 4) {
    //     for (int i = 0; i < blockDim.x; i++) {
    //         printf("%i ", d_out[i]);
    //     }
    //     printf("\n");
    // }
}

void test_reduce(int * d_out, int * d_intermediate, int * d_in, int size)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    /**
	Optimizations mentioned by the course instructor:
	- Processing multiple items per thread, instead of just one
	- Perform first step of the reduction right when you read the items from global to shared memory
	- Take advantage of the fact that warps are synchronous when doing the last steps of the reduction
    */

    const int maxThreadsPerBlock = 4;

    int threads = maxThreadsPerBlock;
    int blocks = size / maxThreadsPerBlock;

    printf("threads = %i\n", threads);
    printf("blocks = %i\n", blocks);

    test_kernel<<<blocks, threads, threads * sizeof(int)>>>
       (d_intermediate, d_in);

    // now we're down to one block left, so reduce it
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;

    test_kernel<<<blocks, threads, threads * sizeof(int)>>>
	    (d_out, d_intermediate);
}


int main(int argc, char **argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem,
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
    }

    const int ARRAY_SIZE = 16;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    // generate the input array on the host
    int h_in[ARRAY_SIZE];
    int sum = 0;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = i + 1;
        sum += h_in[i];
    }

    // print_array(h_in, ARRAY_SIZE);

    // declare GPU memory pointers
    int * d_in, * d_intermediate, * d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); // overallocated
    cudaMalloc((void **) &d_out, sizeof(int));

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // launch the kernel

    printf("Running test reduce\n");
    cudaEventRecord(start, 0);
    for (int i = 0; i < 1/* todo uncomment after testing is done 100 */; i++)
    {
        test_reduce(d_out, d_intermediate, d_in, ARRAY_SIZE);
    }
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    elapsedTime /= 100.0f;      // 100 trials

    // copy back the sum from GPU
    int h_out;
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("average time elapsed: %f\n", elapsedTime);

    printf("\nResults:\n");
    printf("\tserial sum: %i\n", sum);
    printf("\treduce sum: %i\n", h_out);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);

    return 0;
}

// int main_old(int argc, char **argv)
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);
//     if (deviceCount == 0) {
//         fprintf(stderr, "error: no devices supporting CUDA.\n");
//         exit(EXIT_FAILURE);
//     }
//     int dev = 0;
//     cudaSetDevice(dev);

//     cudaDeviceProp devProps;
//     if (cudaGetDeviceProperties(&devProps, dev) == 0)
//     {
//         printf("Using device %d:\n", dev);
//         printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
//                devProps.name, (int)devProps.totalGlobalMem,
//                (int)devProps.major, (int)devProps.minor,
//                (int)devProps.clockRate);
//     }

//     // const int ARRAY_SIZE = 1 << 20;
//     const int ARRAY_SIZE = 12;
//     const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

//     // generate the input array on the host
//     float h_in[ARRAY_SIZE];
//     float sum = 0.0f;
//     for(int i = 0; i < ARRAY_SIZE; i++) {
//         // generate random float in [-1.0f, 1.0f]
//         // When using random() here the result is always -5.174542.
//         // This might be because of how bad this random generator is.
//         // I think that when generating 2^20 random numbers between [-1, 1] their sum should be around 0.
//         h_in[i] = -1.0f + (float)random()/((float)RAND_MAX/2.0f);
//         // h_in[i] = 0.01f; // 0.01 * 2^20 = 10485.76
//         sum += h_in[i];
//     }

//     // declare GPU memory pointers
//     float * d_in, * d_intermediate, * d_out;

//     // allocate GPU memory
//     cudaMalloc((void **) &d_in, ARRAY_BYTES);
//     cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); // overallocated
//     cudaMalloc((void **) &d_out, sizeof(float));

//     // transfer the input array to the GPU
//     cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

//     int whichKernel = 0;
//     if (argc == 2) {
//         whichKernel = atoi(argv[1]);
//     }

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     // launch the kernel
//     switch(whichKernel) {
//     case 0:
//         printf("Running global reduce\n");
//         cudaEventRecord(start, 0);
//         for (int i = 0; i < 100; i++)
//         {
//             reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, false);

//             // global reduce will return wrong sum in this case
//             // is is because it will write the sum to the same memory adress 100 times
//             // to test that out uncomment below and have a look at the partial sums after each iteration
//             // only the first one is correct.
//             // But I don't know why each sum is not just double the previous one,
//             // float h_out;
//             // cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
//             // printf("\tpartial sum: %f\n", h_out);
//         }
//         cudaEventRecord(stop, 0);
//         break;
//     case 1:
//         printf("Running reduce with shared mem\n");
//         cudaEventRecord(start, 0);
//         for (int i = 0; i < 100; i++)
//         {
//             reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
//         }
//         cudaEventRecord(stop, 0);
//         break;
//     case 2:
//         printf("Running optimized reduce with shared mem\n");
//         cudaEventRecord(start, 0);
//         for (int i = 0; i < 1/* todo uncomment after testing is done 100 */; i++)
//         {
//             optimized_reduce(d_out, d_intermediate, d_in, ARRAY_SIZE);
//         }
//         cudaEventRecord(stop, 0);
//         break;
//     default:
//         fprintf(stderr, "error: ran no kernel\n");
//         exit(EXIT_FAILURE);
//     }
//     cudaEventSynchronize(stop);
//     float elapsedTime;
//     cudaEventElapsedTime(&elapsedTime, start, stop);
//     elapsedTime /= 100.0f;      // 100 trials

//     // copy back the sum from GPU
//     float h_out;
//     cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

//     printf("average time elapsed: %f\n", elapsedTime);

//     printf("\nResults:\n");
//     printf("\tserial sum: %f\n", sum);
//     printf("\treduce sum: %f\n", h_out);

//     // free GPU memory allocation
//     cudaFree(d_in);
//     cudaFree(d_intermediate);
//     cudaFree(d_out);

//     return 0;
// }
