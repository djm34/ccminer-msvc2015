#include <memory.h>

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#endif

#include "cuda_helper.h"

#define TPB50 8
#include "cuda_lyra2_vectors.h"

#if __CUDA_ARCH__ == 350 || __CUDA_ARCH__ == 370
#define memshift 4
#else
#define memshift 3
#endif
#if __CUDA_ARCH__ < 500
//#define __ldg4t(x) (*x)  // prevent some issue when global mem isn't sync'd with its cache (but definetely slower)
#define vectype ulonglong4
#define u64type uint64_t
#define memshift 4

#elif __CUDA_ARCH__ == 500
#define vectype ulonglong4
#define u64type uint64_t
#define memshift 3 

#else  
#define vectype uint28
#define u64type uint2
#define memshift 3   

#endif 
#define vectype uint28
#define vectype uint28
#define u64type uint2


__device__  uint2 *DMatrix;
__device__ vectype *DMatrix35;
#define TPB50 32

static __device__ __forceinline__ void Gfunc(uint2 & a, uint2 &b, uint2 &c, uint2 &d)
{
	a += b; d ^= a; d = SWAPUINT2(d);
	c += d; b ^= c; b = ROR2(b, 24);
	a += b; d ^= a; d = ROR2(d, 16);
	c += d; b ^= c; b = ROR2(b, 63);
}

static __device__ __forceinline__ void Gfunc(uint64_t & a, uint64_t &b, uint64_t &c, uint64_t &d)
{

	a += b; d ^= a; d = ROTR64(d, 32);
	c += d; b ^= c; b = ROTR64(b, 24);
	a += b; d ^= a; d = ROTR64(d, 16);
	c += d; b ^= c; b = ROTR64(b, 63);

}

static __device__ __forceinline__ void Gfunc_v35(uint2 & a, uint2 &b, uint2 &c, uint2 &d)
{
	/*
	a += b; d ^= a; d = ROR2(d, 32);
	c += d; b ^= c; b = ROR2(b, 24);
	a += b; d ^= a; d = ROR2(d, 16);
	c += d; b ^= c; b = ROR2(b, 63);
	*/

	a += b; d ^= a; d = SWAPUINT2(d);
	c += d; b ^= c; b = ROR24(b);
	a += b; d ^= a; d = ROR16(d);
	c += d; b ^= c; b = ROR2(b, 63);

}

static __device__ __forceinline__ void Gfunc_v35(unsigned long long & a, unsigned long long &b, unsigned long long &c, unsigned long long &d)
{

	a += b; d ^= a; d = ROTR64(d, 32);
	c += d; b ^= c; b = ROTR64(b, 24);
	a += b; d ^= a; d = ROTR64(d, 16);
	c += d; b ^= c; b = ROTR64(b, 63);

}

static __device__ __forceinline__ void round_lyra_v35(uint28* s)
{

	Gfunc_v35(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc_v35(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc_v35(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc_v35(s[0].w, s[1].w, s[2].w, s[3].w);
	Gfunc_v35(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc_v35(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc_v35(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc_v35(s[0].w, s[1].x, s[2].y, s[3].z);
}

static __device__ __forceinline__ void round_lyra_v35(ulonglong4* s)
{

	Gfunc_v35(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc_v35(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc_v35(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc_v35(s[0].w, s[1].w, s[2].w, s[3].w);

	Gfunc_v35(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc_v35(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc_v35(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc_v35(s[0].w, s[1].x, s[2].y, s[3].z);

}

static __device__ __forceinline__ void round_lyra_v35_ws(vectype &s)
{

	Gfunc_v35(s.x, s.y, s.z, s.w);

	s.y = shuffle2(s.y, threadIdx.x % 4 + 1, 4); //rotate
	s.z = shuffle2(s.z, threadIdx.x % 4 + 2, 4);
	s.w = shuffle2(s.w, threadIdx.x % 4 - 1, 4);

	Gfunc_v35(s.x, s.y, s.z, s.w);

	s.y = shuffle2(s.y, threadIdx.x % 4 - 1, 4); //rotate back
	s.z = shuffle2(s.z, threadIdx.x % 4 - 2, 4);
	s.w = shuffle2(s.w, threadIdx.x % 4 + 1, 4);


}



//#if __CUDA_ARCH__ == 500 || __CUDA_ARCH__ == 350
//#include "cuda_lyra2_vectors.h"

#define Nrow 16
#define Ncol 16
//#define memshift 3



__device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
#if __CUDA_ARCH__ >= 300
	return __shfl(a, b, c);
#else
	extern __shared__ uint2 shared_mem[];

	const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;
	uint32_t *_ptr = (uint32_t*)shared_mem;

	__threadfence_block();
	uint32_t buf = _ptr[thread];

	_ptr[thread] = a;
	__threadfence_block();
	uint32_t result = _ptr[(thread&~(c - 1)) + (b&(c - 1))];

	__threadfence_block();
	_ptr[thread] = buf;

	__threadfence_block();
	return result;
#endif
}

__device__ __forceinline__ uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c)
{
#if __CUDA_ARCH__ >= 300
	return make_uint2(__shfl(a.x, b, c), __shfl(a.y, b, c));
#else
	extern __shared__ uint2 shared_mem[];

	const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;

	__threadfence_block();
	uint2 buf = shared_mem[thread];

	shared_mem[thread] = a;
	__threadfence_block();
	uint2 result = shared_mem[(thread&~(c - 1)) + (b&(c - 1))];

	__threadfence_block();
	shared_mem[thread] = buf;

	__threadfence_block();
	return result;
#endif
}

__device__ __forceinline__ void WarpShuffle3(uint2 &a1, uint2 &a2, uint2 &a3, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
#if __CUDA_ARCH__ >= 300
	a1 = WarpShuffle(a1, b1, c);
	a2 = WarpShuffle(a2, b2, c);
	a3 = WarpShuffle(a3, b3, c);
#else

	extern __shared__ uint2 shared_mem[];

	const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;

	__threadfence_block();
	uint2 buf = shared_mem[thread];

	shared_mem[thread] = a1;
	__threadfence_block();
	a1 = shared_mem[(thread&~(c - 1)) + (b1&(c - 1))];
	__threadfence_block();
	shared_mem[thread] = a2;
	__threadfence_block();
	a2 = shared_mem[(thread&~(c - 1)) + (b2&(c - 1))];
	__threadfence_block();
	shared_mem[thread] = a3;
	__threadfence_block();
	a3 = shared_mem[(thread&~(c - 1)) + (b3&(c - 1))];

	__threadfence_block();
	shared_mem[thread] = buf;
	__threadfence_block();
#endif
}


__device__ __forceinline__ void ST4S(const int row, const int col, const uint2 data[3], const int thread, const int threads)
{

	extern __shared__ uint2 shared_mem[];
	const int s0 = (Ncol * row + col) * memshift;
#pragma unroll
	for (int j = 0; j < 3; j++)
		shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data[j];

}

__device__ __forceinline__ void LD4S2(uint2 res[3], const int row, const int col, const int thread, const int threads, const uint2 * __restrict__ shared_mem)
{

	//	extern __shared__ uint2 shared_mem[];
	const int s0 = (Ncol * row + col) * memshift;

#pragma unroll
	for (int j = 0; j < 3; j++)
		res[j] = shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];

}

__device__ __forceinline__ void ST4S2(const int row, const int col, const uint2 data[3], const int thread, const int threads, uint2 * __restrict__ shared_mem)
{

	//	extern __shared__ uint2 shared_mem[];
	const int s0 = (Ncol * row + col) * memshift;
#pragma unroll
	for (int j = 0; j < 3; j++)
		shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data[j];

}
__device__ __forceinline__ void ST4S2bis(const int row, const int col, const uint2 data[3], const int thread, const int threads, uint2 * __restrict__ shared_mem)
{

	//	extern __shared__ uint2 shared_mem[];
	const int s0 = (Ncol * row + col) * memshift;
#pragma unroll
	for (int j = 0; j < 3; j++)
		shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data[j];

}
__device__ __forceinline__ void LD4S(uint2 res[3], const int row, const int col, const int thread, const int threads)
{

	extern __shared__ uint2 shared_mem[];
	const int s0 = (Ncol * row + col) * memshift;

#pragma unroll
	for (int j = 0; j < 3; j++)
		res[j] = shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];

}

__device__ __forceinline__ uint2 LD4S(const int index)
{
	extern __shared__ uint2 shared_mem[];

	return shared_mem[(index * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
}

__device__ __forceinline__ void ST4S(const int index, const uint2 data)
{
	extern __shared__ uint2 shared_mem[];

	shared_mem[(index * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data;
}

__device__ __forceinline__ void round_lyra(uint2 s[4])
{
	Gfunc(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4);
	Gfunc(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4);
}

static __device__ __forceinline__
void round_lyra(uint2x4* s)
{
	Gfunc(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc(s[0].w, s[1].w, s[2].w, s[3].w);
	Gfunc(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc(s[0].w, s[1].x, s[2].y, s[3].z);
}


static __device__ __forceinline__
void reduceDuplexh2(uint2 state[4], uint32_t thread, const uint32_t threads)
{
#define Nrow 16
#define Ncol 16
	uint2 state1[3];

#if __CUDA_ARCH__ > 500
#pragma unroll
#endif
	for (int i = 0; i < Nrow; i++)
	{
		ST4S(0, Ncol - i - 1, state, thread, threads);

		round_lyra(state);
	}

#pragma unroll 16
	for (int i = 0; i < Nrow; i++)
	{
		LD4S(state1, 0, i, thread, threads);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra(state);

		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];
		ST4S(1, Ncol - i - 1, state1, thread, threads);
	}
}

static __device__ __forceinline__
void reduceDuplexRowSetuph2(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3];
#define Nrow 16
#define Ncol 16
#pragma unroll 1
	for (int i = 0; i < Nrow; i++)
	{
		LD4S(state1, rowIn, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4S(rowOut, Ncol - i - 1, state1, thread, threads);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		ST4S(rowInOut, i, state2, thread, threads);
	}
}

static __device__ __forceinline__
void reduceDuplexRowth2(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], const uint32_t thread, const uint32_t threads)
{
#define Nrow 16
#define Ncol 16
	for (int i = 0; i < Nrow; i++)
	{
		uint2 state1[3], state2[3];

		LD4S(state1, rowIn, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		ST4S(rowInOut, i, state2, thread, threads);

		LD4S(state1, rowOut, i, thread, threads);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4S(rowOut, i, state1, thread, threads);
	}
}

static __device__ __forceinline__
void reduceDuplexRowt_8(const int rowInOut, uint2* state, const uint32_t thread, const uint32_t threads)
{
#define Nrow 16
#define Ncol 16
	uint2 state1[3], state2[3], last[3];

	LD4S(state1, 2, 0, thread, threads);
	LD4S(last, rowInOut, 0, thread, threads);

#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= state1[j] + last[j];

	round_lyra(state);

	//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
	uint2 Data0 = state[0];
	uint2 Data1 = state[1];
	uint2 Data2 = state[2];
	WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

	if (threadIdx.x == 0)
	{
		last[0] ^= Data2;
		last[1] ^= Data0;
		last[2] ^= Data1;
	}
	else {
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == 5)
	{
#pragma unroll
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	for (int i = 1; i < Nrow; i++)
	{
		LD4S(state1, 2, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);
	}

#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}


static __device__ __forceinline__
void reduceDuplexRowt_8_v2h(const int rowIn, const int rowOut, const int rowInOut, uint2* state, const uint32_t thread, const uint32_t threads)
{
#define Nrow 16
#define Ncol 16
	uint2 state1[3], state2[3], last[3];

	LD4S(state1, rowIn, 0, thread, threads);
	LD4S(last, rowInOut, 0, thread, threads);

#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= state1[j] + last[j];

	round_lyra(state);

	//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
	uint2 Data0 = state[0];
	uint2 Data1 = state[1];
	uint2 Data2 = state[2];
	WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

	if (threadIdx.x == 0)
	{
		last[0] ^= Data2;
		last[1] ^= Data0;
		last[2] ^= Data1;
	}
	else {
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == rowOut)
	{
#pragma unroll
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	for (int i = 1; i < Nrow; i++)
	{
		LD4S(state1, rowIn, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);
	}

#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}





static __device__ __forceinline__
void reduceDuplexV5h(uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3];

	const uint32_t ps0 = (memshift * Ncol * 0 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps1 = (memshift * Ncol * 1 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps2 = (memshift * Ncol * 2 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps3 = (memshift * Ncol * 3 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps4 = (memshift * Ncol * 4 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps5 = (memshift * Ncol * 5 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps6 = (memshift * Ncol * 6 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps7 = (memshift * Ncol * 7 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps8 = (memshift * Ncol * 8 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps9 = (memshift * Ncol * 9 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps10 = (memshift * Ncol * 10 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps11 = (memshift * Ncol * 11 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps12 = (memshift * Ncol * 12 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps13 = (memshift * Ncol * 13 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps14 = (memshift * Ncol * 14 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps15 = (memshift * Ncol * 15 * threads + thread)*blockDim.x + threadIdx.x;

	for (int i = 0; i < 16; i++)
	{
		const uint32_t s0 = memshift * Ncol * 0 + (Ncol - 1 - i) * memshift;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			ST4S(s0 + j, state[j]);
		round_lyra(state);
	}

	for (int i = 0; i < 16; i++)
	{
		const uint32_t s0 = memshift * Ncol * 0 + i * memshift;
		const uint32_t s1 = ps1 + (15 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = LD4S(s0 + j);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s1 + j*threads*blockDim.x) = state1[j] ^ state[j];
	}

	// 1, 0, 2
	for (int i = 0; i < 16; i++)
	{
		const uint32_t s0 = memshift * Ncol * 0 + i * memshift;
		const uint32_t s1 = ps1 + i * memshift* threads*blockDim.x;
		const uint32_t s2 = ps2 + (15 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s1 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = LD4S(s0 + j);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s2 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		#pragma unroll
		for (int j = 0; j < 3; j++)
			ST4S(s0 + j, state2[j]);
	}

	// 2, 1, 3
	for (int i = 0; i < 16; i++)
	{
		const uint32_t s1 = ps1 + i * memshift* threads*blockDim.x;
		const uint32_t s2 = ps2 + i * memshift* threads*blockDim.x;
		const uint32_t s3 = ps3 + (15 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s2 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s1 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s3 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		} else  {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s1 + j*threads*blockDim.x) = state2[j];
	}

	// 3, 0, 4
	for (int i = 0; i < 16; i++)
	{
		const uint32_t ls0 = memshift * Ncol * 0 + i * memshift;
		const uint32_t s0 = ps0 + i * memshift* threads*blockDim.x;
		const uint32_t s3 = ps3 + i * memshift* threads*blockDim.x;
		const uint32_t s4 = ps4 + (15 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s3 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = LD4S(ls0 + j);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s4 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		} else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s0 + j*threads*blockDim.x) = state2[j];
	}

	// 4, 3, 5
	for (int i = 0; i < 16; i++)
	{
		const uint32_t s3 = ps3 + i * memshift* threads*blockDim.x;
		const uint32_t s4 = ps4 + i * memshift* threads*blockDim.x;
		const uint32_t s5 = ps5 + (15 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s4 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s3 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s5 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s3 + j*threads*blockDim.x) = state2[j];
	}

	// 5, 2, 6
	for (int i = 0; i < 16; i++)
	{
		const uint32_t s2 = ps2 + i * memshift* threads*blockDim.x;
		const uint32_t s5 = ps5 + i * memshift* threads*blockDim.x;
		const uint32_t s6 = ps6 + (15 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s5 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s2 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s6 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s2 + j*threads*blockDim.x) = state2[j];
	}

	// 6, 1, 7
	for (int i = 0; i < 16; i++)
	{
		const uint32_t s1 = ps1 + i * memshift* threads*blockDim.x;
		const uint32_t s6 = ps6 + i * memshift* threads*blockDim.x;
		const uint32_t s7 = ps7 + (15 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s6 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s1 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s7 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		} else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s1 + j*threads*blockDim.x) = state2[j];
	}

	// 7, 0, 8
	for (int i = 0; i < 16; i++)
	{
		const uint32_t ls0 = memshift * Ncol * 0 + i * memshift;
		const uint32_t s0 = ps0 + i * memshift* threads*blockDim.x;
		const uint32_t s3 = ps7 + i * memshift* threads*blockDim.x;
		const uint32_t s4 = ps8 + (15 - i)*memshift* threads*blockDim.x;
#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s3 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = LD4S(ls0 + j);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s4 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s0 + j*threads*blockDim.x) = state2[j];
	}

	// 8, 3, 9
	for (int i = 0; i < 16; i++)
	{
		const uint32_t s1 = ps3 + i * memshift* threads*blockDim.x;
		const uint32_t s2 = ps8 + i * memshift* threads*blockDim.x;
		const uint32_t s3 = ps9 + (15 - i)*memshift* threads*blockDim.x;
#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s2 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s1 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s3 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s1 + j*threads*blockDim.x) = state2[j];
	}
	// 9, 6, 10
	for (int i = 0; i < 16; i++)
	{
		const uint32_t s1 = ps6 + i * memshift* threads*blockDim.x;
		const uint32_t s2 = ps9 + i * memshift* threads*blockDim.x;
		const uint32_t s3 = ps10 + (15 - i)*memshift* threads*blockDim.x;
#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s2 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s1 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s3 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s1 + j*threads*blockDim.x) = state2[j];
	}

	// 10, 1, 11
	for (int i = 0; i < 16; i++)
	{
		const uint32_t s1 = ps1 + i * memshift* threads*blockDim.x;
		const uint32_t s2 = ps10 + i * memshift* threads*blockDim.x;
		const uint32_t s3 = ps11 + (15 - i)*memshift* threads*blockDim.x;
#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s2 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s1 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s3 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s1 + j*threads*blockDim.x) = state2[j];
	}

	// 11, 4, 12
	for (int i = 0; i < 16; i++)
	{
		const uint32_t s1 = ps4 + i * memshift* threads*blockDim.x;
		const uint32_t s2 = ps11 + i * memshift* threads*blockDim.x;
		const uint32_t s3 = ps12 + (15 - i)*memshift* threads*blockDim.x;
#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s2 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s1 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s3 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s1 + j*threads*blockDim.x) = state2[j];
	}

	// 12, 7, 13
	for (int i = 0; i < 16; i++)
	{
		const uint32_t s1 = ps7 + i * memshift* threads*blockDim.x;
		const uint32_t s2 = ps12 + i * memshift* threads*blockDim.x;
		const uint32_t s3 = ps13 + (15 - i)*memshift* threads*blockDim.x;
#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s2 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s1 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s3 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s1 + j*threads*blockDim.x) = state2[j];
	}

	// 13, 2, 14
	for (int i = 0; i < 16; i++)
	{
		const uint32_t s1 = ps2 + i * memshift* threads*blockDim.x;
		const uint32_t s2 = ps13 + i * memshift* threads*blockDim.x;
		const uint32_t s3 = ps14 + (15 - i)*memshift* threads*blockDim.x;
#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s2 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s1 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s3 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s1 + j*threads*blockDim.x) = state2[j];
	}

	// 14, 5, 15
	for (int i = 0; i < 16; i++)
	{
		const uint32_t s1 = ps5 + i * memshift* threads*blockDim.x;
		const uint32_t s2 = ps14 + i * memshift* threads*blockDim.x;
		const uint32_t s3 = ps15 + (15 - i)*memshift* threads*blockDim.x;
#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s2 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s1 + j*threads*blockDim.x);
#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s3 + j*threads*blockDim.x) = state1[j] ^ state[j];

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s1 + j*threads*blockDim.x) = state2[j];
	}


}

static __device__ __forceinline__
void reduceDuplexRowV50h(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	const uint32_t ps1 = (memshift * Ncol * rowIn*threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps2 = (memshift * Ncol * rowInOut *threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps3 = (memshift * Ncol * rowOut*threads + thread)*blockDim.x + threadIdx.x;

	#pragma unroll 1
	for (int i = 0; i < 16; i++)
	{
		uint2 state1[3], state2[3];

		const uint32_t s1 = ps1 + i*memshift*threads *blockDim.x;
		const uint32_t s2 = ps2 + i*memshift*threads *blockDim.x;
		const uint32_t s3 = ps3 + i*memshift*threads *blockDim.x;

		#pragma unroll
		for (int j = 0; j < 3; j++) {
			state1[j] = *(DMatrix + s1 + j*threads*blockDim.x);
			state2[j] = *(DMatrix + s2 + j*threads*blockDim.x);
		}

		#pragma unroll
		for (int j = 0; j < 3; j++) {
			state1[j] += state2[j];
			state[j] ^= state1[j];
		}

		round_lyra(state);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		} else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		#pragma unroll
		for (int j = 0; j < 3; j++)
		{
			*(DMatrix + s2 + j*threads*blockDim.x) = state2[j];
			*(DMatrix + s3 + j*threads*blockDim.x) ^= state[j];
		}
	}
}

static __device__ __forceinline__
void reduceDuplexRowV50_8h(const int rowInOut, uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	const uint32_t ps1 = (memshift * Ncol * 2*threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps2 = (memshift * Ncol * rowInOut *threads + thread)*blockDim.x + threadIdx.x;
	// const uint32_t ps3 = (memshift * Ncol * 5*threads + thread)*blockDim.x + threadIdx.x;

	uint2 state1[3], last[3];

	#pragma unroll
	for (int j = 0; j < 3; j++) {
		state1[j] = *(DMatrix + ps1 + j*threads*blockDim.x);
		last[j] = *(DMatrix + ps2 + j*threads*blockDim.x);
	}

	#pragma unroll
	for (int j = 0; j < 3; j++) {
		state1[j] += last[j];
		state[j] ^= state1[j];
	}

	round_lyra(state);

	//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
	uint2 Data0 = state[0];
	uint2 Data1 = state[1];
	uint2 Data2 = state[2];
	WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

	if (threadIdx.x == 0)
	{
		last[0] ^= Data2;
		last[1] ^= Data0;
		last[2] ^= Data1;
	} else {
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == 5)
	{
		#pragma unroll
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	for (int i = 1; i < 16; i++)
	{
		const uint32_t s1 = ps1 + i*memshift*threads *blockDim.x;
		const uint32_t s2 = ps2 + i*memshift*threads *blockDim.x;

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= *(DMatrix + s1 + j*threads*blockDim.x) + *(DMatrix + s2 + j*threads*blockDim.x);

		round_lyra(state);
	}


#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];

}


static __device__ __forceinline__
void reduceDuplexRowV50_8_v2h(const int rowIn, const int rowOut,const int rowInOut, uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	const uint32_t ps1 = (memshift * Ncol * rowIn * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps2 = (memshift * Ncol * rowInOut *threads + thread)*blockDim.x + threadIdx.x;
	// const uint32_t ps3 = (memshift * Ncol * 5*threads + thread)*blockDim.x + threadIdx.x;

	uint2 state1[3], last[3];

#pragma unroll
	for (int j = 0; j < 3; j++) {
		state1[j] = *(DMatrix + ps1 + j*threads*blockDim.x);
		last[j] = *(DMatrix + ps2 + j*threads*blockDim.x);
	}

#pragma unroll
	for (int j = 0; j < 3; j++) {
		state1[j] += last[j];
		state[j] ^= state1[j];
	}

	round_lyra(state);

	//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
	uint2 Data0 = state[0];
	uint2 Data1 = state[1];
	uint2 Data2 = state[2];
	WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

	if (threadIdx.x == 0)
	{
		last[0] ^= Data2;
		last[1] ^= Data0;
		last[2] ^= Data1;
	}
	else {
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == rowOut)
	{
#pragma unroll
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	for (int i = 1; i < 16; i++)
	{
		const uint32_t s1 = ps1 + i*memshift*threads *blockDim.x;
		const uint32_t s2 = ps2 + i*memshift*threads *blockDim.x;

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= *(DMatrix + s1 + j*threads*blockDim.x) + *(DMatrix + s2 + j*threads*blockDim.x);

		round_lyra(state);
	}


#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];

}

////////////////////// sm5+ kernels ////////

__device__ __forceinline__ void reduceDuplexRowSetupV2_ws(const int rowIn, const int rowInOut, const int rowOut, vectype state[4], uint32_t thread, uint32_t thread2)
{


	vectype state2[3], state1[3];
	uint32_t ps1 = memshift * (rowIn + 8 * thread2);  // top bottom
	uint32_t ps2 = memshift * (rowInOut + 8 * thread2); //top bottom
	uint32_t ps3 = memshift * (rowOut + 8 * thread2);  //bottom top


#pragma unroll
	for (int j = 0; j < 3; j++)
		state1[j] = __ldg4(&((vectype*)(DMatrix35))[j + ps1]);
#pragma unroll
	for (int j = 0; j < 3; j++)
		state2[j] = __ldg4(&((vectype*)(DMatrix35))[j + ps2]);

	//		 if (thread == 0) printf("thread %d  the matrix content row0 %08llx row1 %08llx    \n", threadIdx.x % 8, state1[0].x, state2[0].x);

	vectype tmp[3];
#pragma unroll
	for (int j = 0; j < 3; j++) {
		tmp[j] = state1[j] + state2[j];
	}

	//////// accross the threads ///////
	//#pragma unroll 
	for (int i = 0; i<8; i++)
	{
		if (threadIdx.x % 8 == 0) {
			for (int j = 0; j < 3; j++)
				state[j] ^= tmp[j];

			round_lyra_v35(state);

		}
		if (i<7) {
			for (int j = 0; j<4; j++)
				state[j] = shuffle_up4(state[j], 1, 8);
			for (int j = 0; j < 3; j++)
				tmp[j] = shuffle_down4(tmp[j], 1, 8);
		}
	}


	//		if (thread == 0) printf("thread %d  the matrix content rowIn %08llx  state %08llx\n", threadIdx.x % 8, state1[0].x, state[0].x);
#pragma unroll 
	for (int j = 0; j<3; j++)
		state1[j] = shuffle_xor4(state1[j], 7, 8);

#pragma unroll 
	for (int j = 0; j < 3; j++)
		state1[j] ^= state[j];

#pragma unroll 
	for (int j = 0; j<3; j++)
		state[j] = shuffle_xor4(state[j], 7, 8);  //reverse order


#pragma unroll 
	for (int j = 0; j<3; j++)
		((vectype*)(DMatrix35))[j + ps3] = state1[j];

	//		if (thread == 0) printf("thread %d before last xor rowOut %08llx  rowInOut %08llx state %08llx\n", threadIdx.x % 8, state1[0].x, state2[0].x, state[0].x);

	((uint2*)state2)[0] ^= ((uint2*)state)[11];
	for (int j = 0; j < 11; j++)
		((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];
#pragma unroll 
	for (int j = 0; j < 3; j++)
		((vectype*)(DMatrix35))[j + ps2] = state2[j];
#pragma unroll 
	for (int j = 0; j<3; j++)
		state[j] = shuffle_xor4(state[j], 7, 8); //reverse back (not entirely necessary)


												 //		if (thread == 0) printf("thread %d  the matrix content rowIn %08llx rowOut %08llx  rowInOut %08llx state %08llx\n", threadIdx.x % 8, (DMatrix35 + ps1)[0].x, state1[0].x, state2[0].x, state[0].x);

}

__device__ __forceinline__ void reduceDuplex_ws(vectype state[4], uint32_t thread, uint32_t thread2)
{


	vectype state1[3], tmp[3];
	uint32_t ps1 = memshift * (8 * thread2);
	uint32_t ps2 = memshift * (1 + 8 * thread2);

#pragma unroll
	for (int j = 0; j < 3; j++)
		state1[j] = __ldg4(&((vectype*)(DMatrix35))[j + ps1]);

#pragma unroll
	for (int j = 0; j < 3; j++)
		tmp[j] = state1[j];
#pragma unroll 
	for (int i = 0; i<8; i++)
	{

		if (threadIdx.x % 8 == 0) {
			for (int j = 0; j < 3; j++)
				state[j] ^= tmp[j];
			round_lyra_v35(state);

		}

		if (i<7) {
			for (int j = 0; j<4; j++)
				state[j] = shuffle_up4(state[j], 1, 8);

			for (int j = 0; j < 3; j++)
				tmp[j] = shuffle_down4(tmp[j], 1, 8);
		}

	}

	for (int j = 0; j<3; j++)
		state1[j] = shuffle_xor4(state1[j], 7, 8); // reverse order 

	for (int j = 0; j < 3; j++)
		state1[j] ^= state[j];


	for (int j = 0; j < 3; j++)
		((vectype*)(DMatrix35))[j + ps2] = state1[j]; //store // matrix end up in top->bottom order 
}

__device__ __forceinline__ void reduceDuplexRowtV2_ws(const int rowIn, int rowInOut, const int rowOut, vectype* state, uint32_t thread, uint32_t thread2)
{

	vectype state1[3], state2[3], tmp[3];
	//	if (thread == 0) printf("rowInOut %d\n",rowInOut);

	rowInOut = __shfl(rowInOut, 0, 8); //broadcast over the 8 lanes
									   //	if (thread == 0) printf("rowInOut %d\n", rowInOut);

	uint32_t ps1 = memshift *(rowIn + 8 * thread2);
	uint32_t ps2 = memshift *(rowInOut + 8 * thread2);
	uint32_t ps3 = memshift *(rowOut + 8 * thread2);

	uint32_t s1 = ps1;
	uint32_t s2 = ps2;
	uint32_t s3 = ps3;

#pragma unroll
	for (int j = 0; j < 3; j++)
		state1[j] = ((vectype*)(DMatrix35))[j + s1];

	if (s1 != s2) {
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = ((vectype*)(DMatrix35))[j + s2];
	}
	else {
		for (int j = 0; j < 3; j++)
			state2[j] = state1[j];
	}

#pragma unroll
	for (int j = 0; j < 3; j++)
		state1[j] += state2[j];
#pragma unroll
	for (int j = 0; j < 3; j++)
		tmp[j] = state1[j];


	///////////////////////////////////////////////

	////////////////////////////////////////////////// 
	//#pragma unroll 

	for (int i = 0; i<8; i++)
	{

		if (threadIdx.x % 8 == 0) {
#pragma unroll
			for (int j = 0; j < 3; j++)
				state[j] ^= tmp[j];
			round_lyra_v35(state);
		}
		if (i<7) {
#pragma unroll
			for (int j = 0; j<4; j++)
				state[j] = shuffle_up4(state[j], 1, 8);
#pragma unroll
			for (int j = 0; j<3; j++)
				tmp[j] = shuffle_down4(tmp[j], 1, 8);
		}
	}

	/////////////////////////////////



	for (int j = 0; j<3; j++)
		state[j] = shuffle_xor4(state[j], 7, 8);  //reverse order


												  ///////////////////////////////////////////////////
	((uint2*)state2)[0] ^= ((uint2*)state)[11];
	for (int j = 0; j < 11; j++)
		((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];

	if (rowInOut != rowOut) {

		for (int j = 0; j < 3; j++)
			((vectype*)(DMatrix35))[j + s2] = state2[j];

		for (int j = 0; j < 3; j++)
			((vectype*)(DMatrix35))[j + s3] ^= state[j];

	}
	else {

		for (int j = 0; j < 3; j++)
			state2[j] ^= state[j];

		for (int j = 0; j < 3; j++)
			((vectype*)(DMatrix35))[j + s2] = state2[j];
	}

	for (int j = 0; j<3; j++)
		state[j] = shuffle_xor4(state[j], 7, 8);  //reverse back



}


static __device__ __forceinline__ void reduceDuplexV3(ulonglong4 state[4], uint32_t thread)
{


	ulonglong4 state1[3];
	uint32_t ps1 = (256 * thread);
	//                     colomn             row
	uint32_t ps2 = (memshift * 7 * 8 + memshift * 1 + 64 * memshift * thread);

#pragma unroll 4
	for (int i = 0; i < 8; i++)
	{
		uint32_t s1 = ps1 + 8 * i *memshift;
		uint32_t s2 = ps2 - 8 * i *memshift;

		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&((ulonglong4*)(DMatrix35))[j + s1]);

		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];
		round_lyra_v35(state);

		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];


		for (int j = 0; j < 3; j++)
			((ulonglong4*)(DMatrix35))[j + s2] = state1[j];

	}

}


static __device__ __forceinline__
void reduceDuplexRowSetupV3(const int rowIn, const int rowInOut, const int rowOut, ulonglong4 state[4], uint32_t thread)
{


	ulonglong4 state2[3], state1[3];

	uint32_t ps1 = (memshift *  rowIn + 64 * memshift * thread);
	uint32_t ps2 = (memshift * rowInOut + 64 * memshift* thread);
	uint32_t ps3 = (8 * memshift * 7 + memshift *  rowOut + 64 * memshift * thread);
	/*
	uint32_t ps1 = (256 * thread);
	uint32_t ps2 = (256 * thread);
	uint32_t ps3 = (256 * thread);
	*/
#pragma nounroll 
	for (int i = 0; i < 8; i++)
	{
		uint32_t s1 = ps1 + 8 * i*memshift;
		uint32_t s2 = ps2 + 8 * i*memshift;
		uint32_t s3 = ps3 - 8 * i*memshift;

		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&((ulonglong4*)(DMatrix35))[j + s1]);
		if (s1 != s2)
			for (int j = 0; j < 3; j++)
				state2[j] = __ldg4(&((ulonglong4*)(DMatrix35))[j + s2]);
		else
			for (int j = 0; j < 3; j++)
				state2[j] = state1[j];

		for (int j = 0; j < 3; j++) {
			ulonglong4 tmp = state1[j] + state2[j];
			state[j] ^= tmp;
		}


		round_lyra_v35(state);

		for (int j = 0; j < 3; j++) {
			state1[j] ^= state[j];
			((ulonglong4*)(DMatrix35))[j + s3] = state1[j];
		}

		((uint2*)state2)[0] ^= ((uint2*)state)[11];
		for (int j = 0; j < 11; j++)
			((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];



		for (int j = 0; j < 3; j++)
			((ulonglong4*)(DMatrix35))[j + s2] = state2[j];

	}


}

static __device__ __forceinline__
void reduceDuplexRowtV3(const int rowIn, const int rowInOut, const int rowOut, ulonglong4* state, uint32_t thread)
{

	ulonglong4 state1[3], state2[3];
	uint32_t ps1 = (memshift * rowIn + 64 * memshift * thread);
	uint32_t ps2 = (memshift * rowInOut + 64 * memshift * thread);
	uint32_t ps3 = (memshift * rowOut + 64 * memshift * thread);

#pragma nounroll 
	for (int i = 0; i < 8; i++)
	{
		uint32_t s1 = ps1 + 8 * i*memshift;
		uint32_t s2 = ps2 + 8 * i*memshift;
		uint32_t s3 = ps3 + 8 * i*memshift;


		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&((ulonglong4*)(DMatrix35))[j + s1]);

		//if (s1!=s2)
		for (int j = 0; j < 3; j++)
			state2[j] = __ldg4(&((ulonglong4*)(DMatrix35))[j + s2]);
		/*else
		for (int j = 0; j < 3; j++)
		state2[j] = state1[j];
		*/
		for (int j = 0; j < 3; j++)
			state1[j] += state2[j];

		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];


		round_lyra_v35(state);

		((uint2*)state2)[0] ^= ((uint2*)state)[11];
		for (int j = 0; j < 11; j++)
			((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];

		if (rowInOut != rowOut) {

			for (int j = 0; j < 3; j++)
				((ulonglong4*)(DMatrix35))[j + s2] = state2[j];

			for (int j = 0; j < 3; j++)
				((ulonglong4*)(DMatrix35))[j + s3] ^= state[j];

		}
		else {

			for (int j = 0; j < 3; j++)
				state2[j] ^= state[j];

			for (int j = 0; j < 3; j++)
				((ulonglong4*)(DMatrix35))[j + s2] = state2[j];
		}






	}
}

static __device__ __forceinline__ void reduceDuplex_ws2(vectype &state, uint32_t thread, vectype *__restrict__ tmp)
{


	vectype state1;

	uint32_t ps2 = (memshift * (Ncol - 1) + memshift * Ncol + Nrow * Ncol * memshift * thread);

#pragma unroll 4
	for (int i = 0; i < Ncol; i++)
	{
		uint32_t s2 = ps2 - i*memshift;

		state1 = tmp[i]; //tmp[i];
		state.x ^= state1.x;
		state.y ^= state1.y;
		state.z ^= state1.z;

		round_lyra_v35_ws(state);
		state1 ^= state;
		tmp[i] = state1;
		(DMatrix35 + s2)[threadIdx.x % 4] = state1;

	}

}

static __device__ __forceinline__ void reduceDuplexRowSetup_ws2_pass1(const int rowIn, const int rowInOut, const int rowOut, vectype &state, uint32_t thread, vectype *__restrict__ temp)
{

	uint32_t laneId = threadIdx.x & 3;
	vectype state2, state1;
	vectype rstate;

	uint32_t ps2 = (memshift * Ncol * rowInOut + Nrow * Ncol * memshift * thread);
	uint32_t ps3 = (memshift * (Ncol - 1) + memshift * Ncol * rowOut + Nrow * Ncol * memshift * thread);

	//#pragma unroll 1
	for (int i = 0; i < Ncol; i++)
	{

		uint32_t s2 = ps2 + i*memshift;
		uint32_t s3 = ps3 - i*memshift;


		state1 = temp[Ncol - 1 - i];
		state2 = __ldg4t(&(DMatrix35 + s2)[laneId]);

		vectype tmp = state1 + state2;
		state.x ^= tmp.x;
		state.y ^= tmp.y;
		state.z ^= tmp.z;



		round_lyra_v35_ws(state);


		state1 ^= state;
		(DMatrix35 + s3)[laneId] = state1;
		temp[Ncol - 1 - i] = state1;

		rstate.x = shuffle2(state.x, laneId - 1, 4);
		rstate.y = shuffle2(state.y, laneId - 1, 4);
		rstate.z = shuffle2(state.z, laneId - 1, 4);

		if (laneId == 0)
		{
			u64type tmp2 = rstate.z;
			rstate.z = rstate.y;
			rstate.y = rstate.x;
			rstate.x = tmp2;
		}
		state2 ^= rstate;
		(DMatrix35 + s2)[laneId] = state2;

	}


}

static __device__ __forceinline__ void reduceDuplexRowSetup_ws2_pass2(const int rowIn, const int rowInOut, const int rowOut, vectype &state, uint32_t thread, vectype *__restrict__ temp)
{


	vectype state2, state1;
	vectype rstate;

	uint32_t ps2 = (memshift * Ncol * rowInOut + Nrow * Ncol * memshift * thread);
	uint32_t ps3 = (memshift * (Ncol - 1) + memshift * Ncol * rowOut + Nrow * Ncol * memshift * thread);
	uint32_t laneId = threadIdx.x & 3;
	//#pragma unroll 1
	for (int i = 0; i < Ncol; i++)
	{

		uint32_t s2 = ps2 + i*memshift;
		uint32_t s3 = ps3 - i*memshift;

		state1 = temp[i];
		state2 = __ldg4t(&(DMatrix35 + s2)[laneId]);

		vectype tmp = state1 + state2;
		state.x ^= tmp.x;
		state.y ^= tmp.y;
		state.z ^= tmp.z;



		round_lyra_v35_ws(state);


		state1 ^= state;
		(DMatrix35 + s3)[laneId] = state1;
		temp[i] = state1;

		rstate.x = shuffle2(state.x, laneId - 1, 4);
		rstate.y = shuffle2(state.y, laneId - 1, 4);
		rstate.z = shuffle2(state.z, laneId - 1, 4);

		if (threadIdx.x % 4 == 0)
		{
			u64type tmp2 = rstate.z;
			rstate.z = rstate.y;
			rstate.y = rstate.x;
			rstate.x = tmp2;
		}
		state2 ^= rstate;
		(DMatrix35 + s2)[laneId] = state2;

	}


}

static __device__ __forceinline__ void reduceDuplexRow_ws2(const int rowIn, const int rowInOut, const int rowOut, vectype &state, uint32_t thread, vectype * __restrict__ temp)
{

	vectype state1, state2, rstate;
	uint32_t ps2 = (memshift * Ncol * rowInOut + Nrow * Ncol * memshift * thread);
	uint32_t ps3 = (memshift * Ncol * rowOut + Nrow * Ncol * memshift * thread);
	uint32_t laneId = threadIdx.x & 3;
	//#pragma unroll 1
	for (int i = 0; i < Ncol; i++)
	{

		uint32_t s2 = ps2 + i*memshift;
		uint32_t s3 = ps3 + i*memshift;

		state1 = temp[Ncol - 1 - i];

		if (rowIn != rowInOut)
			state2 = __ldg4t(&(DMatrix35 + s2)[laneId]);
		else {
			state2 = state1;

		}
		state1 += state2;


		state.x ^= state1.x;
		state.y ^= state1.y;
		state.z ^= state1.z;

		round_lyra_v35_ws(state);

		rstate.x = shuffle2(state.x, laneId - 1, 4);
		rstate.y = shuffle2(state.y, laneId - 1, 4);
		rstate.z = shuffle2(state.z, laneId - 1, 4);

		if (laneId == 0)
		{
			u64type tmp = rstate.z;
			rstate.z = rstate.y;
			rstate.y = rstate.x;
			rstate.x = tmp;
		}
		state2 ^= rstate;


		if (rowInOut != rowOut) {

			(DMatrix35 + s2)[laneId] = state2;
			temp[Ncol - 1 - i] = __ldg4t(&(DMatrix35 + s3)[laneId]);
			temp[Ncol - 1 - i] ^= state;
			(DMatrix35 + s3)[laneId] = temp[Ncol - 1 - i];

		}
		else {

			state2 ^= state;
			temp[Ncol - 1 - i] = state2;
			(DMatrix35 + s2)[laneId] = state2;
		}


	}
}

static __device__ __forceinline__ void reduceDuplexRow_ws2_v2(const int rowIn, const int rowInOut, const int rowOut, vectype &state, uint32_t thread, vectype * __restrict__ temp)
{

	vectype state1, state2, rstate;
	uint32_t ps2 = (memshift * Ncol * rowInOut + Nrow * Ncol * memshift * thread);
	uint32_t ps3 = (memshift * Ncol * rowOut + Nrow * Ncol * memshift * thread);
	uint32_t laneId = threadIdx.x & 3;
	//#pragma unroll 1
	for (int i = 0; i < Ncol; i++)
	{

		uint32_t s2 = ps2 + i*memshift;
		uint32_t s3 = ps3 + i*memshift;

		state1 = temp[Ncol - 1 - i];

		if (rowIn != rowInOut)
			state2 = __ldg4t(&(DMatrix35 + s2)[laneId]);
		else {
			state2 = state1;

		}
		state1 += state2;


		state.x ^= state1.x;
		state.y ^= state1.y;
		state.z ^= state1.z;

		round_lyra_v35_ws(state);

		if (i == 0) {

			rstate.x = shuffle2(state.x, laneId - 1, 4);
			rstate.y = shuffle2(state.y, laneId - 1, 4);
			rstate.z = shuffle2(state.z, laneId - 1, 4);

			if (laneId == 0)
			{
				u64type tmp = rstate.z;
				rstate.z = rstate.y;
				rstate.y = rstate.x;
				rstate.x = tmp;
			}
			state2 ^= rstate;


			if (rowInOut != rowOut) {
				temp[Ncol - 1 - i] = state2;
				/*
				(DMatrix35 + s2)[laneId] = state2;
				temp[7 - i] = __ldg4t(&(DMatrix35 + s3)[laneId]);
				temp[7 - i] ^= state;
				(DMatrix35 + s3)[laneId] = temp[7 - i];
				*/
			}
			else {

				state2 ^= state;
				temp[Ncol - 1 - i] = state2;
				//			(DMatrix35 + s2)[laneId] = state2;
			}

		}
	}
	//	temp[0] = __ldg4t(&(DMatrix35 + ps2)[laneId]);
}



