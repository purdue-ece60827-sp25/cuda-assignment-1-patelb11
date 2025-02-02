
#include "cudaLib.cuh"	

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	
	//get threadId
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	//do calculation if threadId is valid (less than size)
	if(threadId < size){
		y[threadId] = x[threadId] * scale + y[threadId];
	}
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	// generate the two array for calculation (A*X+Y) on Host Side
	float *x, *y, *y_copy;

	x = (float *)malloc(vectorSize * sizeof(float));
	y = (float *)malloc(vectorSize * sizeof(float));
	y_copy = (float *)malloc(vectorSize * sizeof(float));

	//check if malloc worked 
	if(x == NULL || y == NULL || y_copy == NULL){
		std::cout << "Malloc did not work! \n"; 
		return -1;
	}

	// fill the vectors with their respective values 
	vectorInit(x, vectorSize);
	vectorInit(y, vectorSize);
	std::memcpy(y_copy, y, vectorSize * sizeof(float));
	// float a = rand() % 5 + 1; 
	float a = 2.3565; 

	//allocate device memory
	float *x_device, *y_device;
	cudaMalloc(&x_device, vectorSize * sizeof(float));
	cudaMalloc(&y_device, vectorSize * sizeof(float));

	//copy data from host to device 
	cudaMemcpy(x_device, x, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(y_device, y, vectorSize * sizeof(float), cudaMemcpyHostToDevice);

	//launch the saxpy_gpu Kernal 
	int threads_per_threadblock = 128;
	int num_threadblocks = (vectorSize + threads_per_threadblock - 1) / threads_per_threadblock; //ceiling of vectorSize/threads_per_threadblock
	saxpy_gpu<<<num_threadblocks,threads_per_threadblock>>>(x_device, y_device, a, vectorSize);

	//copy result of data back to host 
	cudaMemcpy(y, y_device, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);

	//free all device memory 
	cudaFree(x_device);
	cudaFree(y_device);

	//check if the SAXPY calculation is correct 
	// int errorCount = verifyVector(x, y_copy, y, a, vectorSize);
	int errorCount = 0; 
	for(int i = 0; i < vectorSize; i++){
		if((a*x[i] + y_copy[i]) - y[i] > .0001){
			errorCount++; 
			std::cout << "Idx " << i << " expected " << a*x[i] + y_copy[i]
					<< " found " << y[i] << "\n";
		}
	}

	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	//free all host memory 
	free(x);
	free(y);
	free(y_copy);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

//this function you are generating sampleSize number of points and seeing if they are a hit 
//assume each point is a individual thread 
__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	
	//get the threadId 
	uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

	//check if threadId is valid
	if(tid >= pSumSize){
		return; 
	}

	//create curand state
	curandState state;
    curand_init(clock64(), tid, 0, &state);

	//generate sampleSize points and check if they are a 'hit'
	uint64_t hit_count = 0;
	for(uint64_t i = 0; i < sampleSize; i++){
		float x = curand_uniform(&state);
		float y = curand_uniform(&state);
		if(uint64_t(x*x + y*y) == 0){
			hit_count++;
		}
	}

	//take the number of hits and put into pSums
	pSums[tid] = hit_count;
}

//given reduceSize which gives us the unmber of elements in pSum to combine and output to totals 
__global__ 
void reduceCounts(uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	
	//get the threadId 
	uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

	//check if tid is not too large compared to the size of total 
	uint64_t size_total = (pSumSize + reduceSize - 1)/reduceSize;
	if(tid >= size_total){
		return; 
	}

	//sum the values in pSums
	uint64_t sum = 0;
	for(uint64_t i = 0; i < reduceSize; i++){

		if(i+tid*reduceSize >= pSumSize){
			continue; 
		}

		sum += pSums[i+tid*reduceSize];
	}

	//place the sum into the total array 
	totals[tid] = sum; 
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	// create the variables for the device and host
	uint64_t *totals_host = new uint64_t[reduceThreadCount];

	uint64_t *pSums_device, *totals_device;
	cudaMalloc(&pSums_device, generateThreadCount * sizeof(uint64_t));
	cudaMalloc(&totals_device, reduceThreadCount * sizeof(uint64_t));

	// launch generate points kernal 
	uint64_t threads_per_threadblock = 256;
	uint64_t num_threadblocks = (generateThreadCount + threads_per_threadblock - 1)/ threads_per_threadblock; 

	generatePoints<<<num_threadblocks, threads_per_threadblock>>>(pSums_device, generateThreadCount, sampleSize);
    cudaDeviceSynchronize();

    // launch reduce points
	threads_per_threadblock = 256;
	num_threadblocks = (reduceThreadCount + threads_per_threadblock - 1)/ threads_per_threadblock; 

	reduceCounts<<<num_threadblocks, threads_per_threadblock>>>(pSums_device, totals_device, generateThreadCount, reduceSize);
    cudaDeviceSynchronize();

	//now copy over the total array from device back to host 
	cudaMemcpy(totals_host, totals_device, reduceThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	//now iterate through total array and get total number of hits 
	uint64_t total_hits = 0;
	for(uint64_t i = 0; i < reduceThreadCount; i++){
		total_hits += totals_host[i];
	}

	//calcuate the estimate of pi using total hits
	approxPi = float(total_hits) / (generateThreadCount * sampleSize);
	approxPi = approxPi * 4.0f;

	cudaFree(pSums_device);
	cudaFree(totals_device);
	delete[] totals_host;

	return approxPi;
}
