#include "CudaAtomicList.h"
#include "CudaAtomicList.cuh"

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

static __constant__ void *_LIST_BUF[1];
static __constant__ unsigned int *_LIST_SIZE[1];


__device__ void atomicListAdd(void *info, unsigned int size)
{
	unsigned int count = atomicAdd(_LIST_SIZE[0], 1);

	unsigned char *ptr = (unsigned char *)(_LIST_BUF[0]) + count * size;

	memcpy(ptr, info, size);
}

static cudaError_t setListPtr(void *ptr, unsigned int *numResults)
{
	cudaError_t err = cudaMemcpyToSymbol(_LIST_BUF, &ptr, sizeof(void *));

	if(err) {
		printf("cudaatomiclist setListPtr: cudaMemcpyToSymbol list_buf error!");
		return err;
	}

	err = cudaMemcpyToSymbol(_LIST_SIZE, &numResults, sizeof(unsigned int *));
        if (err) {
		printf("cudaatomiclist setListPtr: cudaMemcpyToSymbol list_size error!");
		return err;
	}
	return err;
}


cudaError_t CudaAtomicList::init(unsigned int itemSize, unsigned int maxItems)
{
	_itemSize = itemSize;

	// The number of results found in the most recent kernel run
	_countHostPtr = NULL;
	cudaError_t err = cudaHostAlloc(&_countHostPtr, sizeof(unsigned int), cudaHostAllocMapped);
	if(err) {
		printf("cudaAtomicList::init: cudaHostAlloc countHostPtr error!");
		goto end;
	}

	// Number of items in the list
	_countDevPtr = NULL;
	err = cudaHostGetDevicePointer(&_countDevPtr, _countHostPtr, 0);
	if(err) {
		printf("cudaAtomicList::init: cudaHostGetDevicePointer countDevicePte error!");
		goto end;
	}
	*_countHostPtr = 0;

	// Storage for results data
	_hostPtr = NULL;
	err = cudaHostAlloc(&_hostPtr, itemSize * maxItems, cudaHostAllocMapped);
	if(err) {
		printf("cudaAtomicList::init: cudaHostAlloc hostPtr error!");
		goto end;
	}

	// Storage for results data (device to host pointer)
	_devPtr = NULL;
	err = cudaHostGetDevicePointer(&_devPtr, _hostPtr, 0);

	if(err) {
		printf("cudaAtomicList::init: cudaHostGetDevicePointer devPtr error!");
		goto end;
	}

	err = setListPtr(_devPtr, _countDevPtr);

end:
	if(err) {
		cudaFreeHost(_countHostPtr);

		cudaFree(_countDevPtr);

		cudaFreeHost(_hostPtr);

		cudaFree(_devPtr);
	}

	return err;
}

unsigned int CudaAtomicList::size()
{
	return *_countHostPtr;
}

void CudaAtomicList::clear()
{
	*_countHostPtr = 0;
}

unsigned int CudaAtomicList::read(void *ptr, unsigned int count)
{
	if(count >= *_countHostPtr) {
		count = *_countHostPtr;
	}

	memcpy(ptr, _hostPtr, count * _itemSize);

	return count;
}

void CudaAtomicList::cleanup()
{
	cudaFreeHost(_countHostPtr);

	cudaFree(_countDevPtr);

	cudaFreeHost(_hostPtr);

	cudaFree(_devPtr);
}
