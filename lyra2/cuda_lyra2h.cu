/**
 * Lyra2 (v1) cuda implementation based on djm34 work
 * tpruvot@github 2015, Nanashi 08/2016 (from 1.8-r2)
 * Lyra2h implentation for HPP coin based on all the previous
 * djm34 2017
 **/

#include <stdio.h>
#include <memory.h>

#define TPB52 8
#define TPB30 160
#define TPB20 160


//#include "cuda_lyra2h_sm2.cuh"
#include "cuda_lyra2h_sm5.cuh"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#endif
static uint32_t *h_GNonces[16]; // this need to get fixed as the rest of that routine
static uint32_t *d_GNonces[16];

__constant__ uint32_t pTarget[8];

__constant__ static uint2 blake2b_IV_sm2[8] = {
	{ 0xf3bcc908, 0x6a09e667 },
	{ 0x84caa73b, 0xbb67ae85 },
	{ 0xfe94f82b, 0x3c6ef372 },
	{ 0x5f1d36f1, 0xa54ff53a },
	{ 0xade682d1, 0x510e527f },
	{ 0x2b3e6c1f, 0x9b05688c },
	{ 0xfb41bd6b, 0x1f83d9ab },
	{ 0x137e2179, 0x5be0cd19 }
};

#define reduceDuplexRow(rowIn, rowInOut, rowOut) { \
	for (int i = 0; i < 16; i++) { \
		for (int j = 0; j < 12; j++) \
			state[j] ^= Matrix[12 * i + j][rowIn] + Matrix[12 * i + j][rowInOut]; \
		round_lyra_sm2(state); \
		for (int j = 0; j < 12; j++) \
			Matrix[j + 12 * i][rowOut] ^= state[j]; \
		Matrix[0 + 12 * i][rowInOut] ^= state[11]; \
		Matrix[1 + 12 * i][rowInOut] ^= state[0]; \
		Matrix[2 + 12 * i][rowInOut] ^= state[1]; \
		Matrix[3 + 12 * i][rowInOut] ^= state[2]; \
		Matrix[4 + 12 * i][rowInOut] ^= state[3]; \
		Matrix[5 + 12 * i][rowInOut] ^= state[4]; \
		Matrix[6 + 12 * i][rowInOut] ^= state[5]; \
		Matrix[7 + 12 * i][rowInOut] ^= state[6]; \
		Matrix[8 + 12 * i][rowInOut] ^= state[7]; \
		Matrix[9 + 12 * i][rowInOut] ^= state[8]; \
		Matrix[10+ 12 * i][rowInOut] ^= state[9]; \
		Matrix[11+ 12 * i][rowInOut] ^= state[10]; \
	} \
  }

#define absorbblock(in)  { \
	state[0] ^= Matrix[0][in]; \
	state[1] ^= Matrix[1][in]; \
	state[2] ^= Matrix[2][in]; \
	state[3] ^= Matrix[3][in]; \
	state[4] ^= Matrix[4][in]; \
	state[5] ^= Matrix[5][in]; \
	state[6] ^= Matrix[6][in]; \
	state[7] ^= Matrix[7][in]; \
	state[8] ^= Matrix[8][in]; \
	state[9] ^= Matrix[9][in]; \
	state[10] ^= Matrix[10][in]; \
	state[11] ^= Matrix[11][in]; \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
  }


__device__ __forceinline__
static void round_lyra_sm2(uint2 *s)
{
	Gfunc(s[0], s[4], s[8], s[12]);
	Gfunc(s[1], s[5], s[9], s[13]);
	Gfunc(s[2], s[6], s[10], s[14]);
	Gfunc(s[3], s[7], s[11], s[15]);
	Gfunc(s[0], s[5], s[10], s[15]);
	Gfunc(s[1], s[6], s[11], s[12]);
	Gfunc(s[2], s[7], s[8], s[13]);
	Gfunc(s[3], s[4], s[9], s[14]);
}

__device__ __forceinline__
void reduceDuplexRowSetuph(const int rowIn, const int rowInOut, const int rowOut, uint2 state[16], uint2 Matrix[192][16])
{
#if __CUDA_ARCH__ > 500
#pragma unroll
#endif
	for (int i = 0; i < 16; i++)
	{
#pragma unroll
		for (int j = 0; j < 12; j++)
			state[j] ^= Matrix[12 * i + j][rowIn] + Matrix[12 * i + j][rowInOut];

		round_lyra_sm2(state);

#pragma unroll
		for (int j = 0; j < 12; j++)
			Matrix[j + 180 - 12 * i][rowOut] = Matrix[12 * i + j][rowIn] ^ state[j];

		Matrix[0 + 12 * i][rowInOut] ^= state[11];
		Matrix[1 + 12 * i][rowInOut] ^= state[0];
		Matrix[2 + 12 * i][rowInOut] ^= state[1];
		Matrix[3 + 12 * i][rowInOut] ^= state[2];
		Matrix[4 + 12 * i][rowInOut] ^= state[3];
		Matrix[5 + 12 * i][rowInOut] ^= state[4];
		Matrix[6 + 12 * i][rowInOut] ^= state[5];
		Matrix[7 + 12 * i][rowInOut] ^= state[6];
		Matrix[8 + 12 * i][rowInOut] ^= state[7];
		Matrix[9 + 12 * i][rowInOut] ^= state[8];
		Matrix[10 + 12 * i][rowInOut] ^= state[9];
		Matrix[11 + 12 * i][rowInOut] ^= state[10];
	}
}



__global__ __launch_bounds__(256, 1)
void lyra2h_gpu_hash_32_sm2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *resNonces)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint2 Mask[8] = {
		{ 0x00000020, 0x00000000 },{ 0x00000020, 0x00000000 },
		{ 0x00000020, 0x00000000 },{ 0x00000010, 0x00000000 },
		{ 0x00000010, 0x00000000 },{ 0x00000010, 0x00000000 },
		{ 0x00000080, 0x00000000 },{ 0x00000000, 0x01000000 }
	};
	if (thread < threads)
	{
		uint2 state[16];

#pragma unroll
		for (int i = 0; i<4; i++) {
			LOHI(state[i].x, state[i].y, g_hash[threads*i + thread]);
		} //password

#pragma unroll
		for (int i = 0; i<4; i++) {
			state[i + 4] = state[i];
		} //salt

#pragma unroll
		for (int i = 0; i<8; i++) {
			state[i + 8] = blake2b_IV_sm2[i];
		}

		// blake2blyra x2
		//#pragma unroll 24
		for (int i = 0; i<12; i++) {
			round_lyra_sm2(state);
		}

		for (int i = 0; i<8; i++)
			state[i] ^= Mask[i];


		for (int i = 0; i<12; i++) {
			round_lyra_sm2(state);
		}


		uint2 Matrix[192][16]; // not cool

							 // reducedSqueezeRow0
#pragma unroll 16
		for (int i = 0; i < 16; i++)
		{
#pragma unroll 12
			for (int j = 0; j<12; j++) {
				Matrix[j + 180 - 12 * i][0] = state[j];
			}
			round_lyra_sm2(state);
		}

		// reducedSqueezeRow1
#pragma unroll 16
		for (int i = 0; i < 16; i++)
		{
#pragma unroll 12
			for (int j = 0; j<12; j++) {
				state[j] ^= Matrix[j + 12 * i][0];
			}
			round_lyra_sm2(state);
#pragma unroll 12
			for (int j = 0; j<12; j++) {
				Matrix[j + 180 - 12 * i][1] = Matrix[j + 12 * i][0] ^ state[j];
			}
		}

		reduceDuplexRowSetuph(1, 0, 2, state, Matrix);
		reduceDuplexRowSetuph(2, 1, 3, state, Matrix);
		reduceDuplexRowSetuph(3, 0, 4, state, Matrix);
		reduceDuplexRowSetuph(4, 3, 5, state, Matrix);
		reduceDuplexRowSetuph(5, 2, 6, state, Matrix);
		reduceDuplexRowSetuph(6, 1, 7, state, Matrix);
		reduceDuplexRowSetuph(7, 0, 8, state, Matrix);
		reduceDuplexRowSetuph(8, 3, 9, state, Matrix);
		reduceDuplexRowSetuph(9, 6, 10, state, Matrix);
		reduceDuplexRowSetuph(10, 1, 11, state, Matrix);
		reduceDuplexRowSetuph(11, 4, 12, state, Matrix);
		reduceDuplexRowSetuph(12, 7, 13, state, Matrix);
		reduceDuplexRowSetuph(13, 2, 14, state, Matrix);
		reduceDuplexRowSetuph(14, 5, 15, state, Matrix);

		uint32_t rowa;
		uint32_t prev = 15;
		uint32_t iterator = 0;

		for (int tau=0;tau<8;tau++) {
			for (uint32_t i = 0; i<16; i++) {
				rowa = state[0].x & 15;
				reduceDuplexRow(prev, rowa, iterator);
				prev = iterator;
				iterator = (iterator + 7) & 15;
			}
			for (uint32_t i = 0; i<16; i++) {
				rowa = state[0].x & 15;
				reduceDuplexRow(prev, rowa, iterator);
				prev = iterator;
				iterator = (iterator - 1) & 15;
			}
		}



		absorbblock(rowa);
		uint32_t nonce = startNounce + thread;
		if (((uint64_t*)state)[3] <= ((uint64_t*)pTarget)[3]) {
			atomicMin(&resNonces[1], resNonces[0]);
			atomicMin(&resNonces[0], nonce);
		}
	} //thread
}


__global__ __launch_bounds__(64, 1)
void lyra2h_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint2x4 Mask[2] = {
		0x00000020UL, 0x00000000UL, 0x00000020UL, 0x00000000UL,
		0x00000020UL, 0x00000000UL, 0x00000010UL, 0x00000000UL,
		0x00000010UL, 0x00000000UL, 0x00000010UL, 0x00000000UL,
		0x00000080UL, 0x00000000UL, 0x00000000UL, 0x01000000UL
	};
	const uint2x4 blake2b_IV[2] = {
	0xf3bcc908lu, 0x6a09e667lu,
	0x84caa73blu, 0xbb67ae85lu,
	0xfe94f82blu, 0x3c6ef372lu,
	0x5f1d36f1lu, 0xa54ff53alu,
	0xade682d1lu, 0x510e527flu,
	0x2b3e6c1flu, 0x9b05688clu,
	0xfb41bd6blu, 0x1f83d9ablu,
	0x137e2179lu, 0x5be0cd19lu
    };
	if (thread < threads)
	{
		uint2x4 state[4];

		state[0].x = state[1].x = __ldg(&g_hash[thread + threads * 0]);
		state[0].y = state[1].y = __ldg(&g_hash[thread + threads * 1]);
		state[0].z = state[1].z = __ldg(&g_hash[thread + threads * 2]);
		state[0].w = state[1].w = __ldg(&g_hash[thread + threads * 3]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i<12; i++)
			round_lyra(state); //because 12 is not enough

		state[0] ^= Mask[0];
		state[1] ^= Mask[1];


		for (int i = 0; i<12; i++)
			round_lyra(state); //because 12 is not enough

		((uint2x4*)DMatrix)[threads * 0 + thread] = state[0];
		((uint2x4*)DMatrix)[threads * 1 + thread] = state[1];
		((uint2x4*)DMatrix)[threads * 2 + thread] = state[2];
		((uint2x4*)DMatrix)[threads * 3 + thread] = state[3];
	}
}

__global__
//__launch_bounds__(TPB52, 1)
void lyra2h_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;

	if (thread < threads)
	{
		uint2 state[4];
		state[0] = __ldg(&DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x]);
		state[1] = __ldg(&DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x]);
		state[2] = __ldg(&DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x]);
		state[3] = __ldg(&DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x]);

		reduceDuplexh2(state, thread, threads);
		reduceDuplexRowSetuph2(1, 0, 2, state, thread, threads);
		reduceDuplexRowSetuph2(2, 1, 3, state, thread, threads);
		reduceDuplexRowSetuph2(3, 0, 4, state, thread, threads);
		reduceDuplexRowSetuph2(4, 3, 5, state, thread, threads);
		reduceDuplexRowSetuph2(5, 2, 6, state, thread, threads);
		reduceDuplexRowSetuph2(6, 1, 7, state, thread, threads);
		reduceDuplexRowSetuph2(7, 0, 8, state, thread, threads);
		reduceDuplexRowSetuph2(8, 3, 9, state, thread, threads);
		reduceDuplexRowSetuph2(9, 6, 10, state, thread, threads);
		reduceDuplexRowSetuph2(10, 1, 11, state, thread, threads);
		reduceDuplexRowSetuph2(11, 4, 12, state, thread, threads);
		reduceDuplexRowSetuph2(12, 7, 13, state, thread, threads);
		reduceDuplexRowSetuph2(13, 2, 14, state, thread, threads);
		reduceDuplexRowSetuph2(14, 5, 15, state, thread, threads);

		uint32_t rowa; // = WarpShuffle(state[0].x, 0, 4) & 7;

		uint32_t prev = 15;
		uint32_t iterator = 0;


for (uint32_t tau=0;tau<7;tau++) {
		
		for (uint32_t i = 0; i<16; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 15;
			reduceDuplexRowth2(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 7) & 15;
		}

		for (uint32_t i = 0; i<16; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 15;
			reduceDuplexRowth2(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 15;
		}
}

		for (uint32_t i = 0; i<16; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 15;
			reduceDuplexRowth2(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 7) & 15;
		}

		for (uint32_t i = 0; i<15; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 15;
			reduceDuplexRowth2(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 15;
		}

//}
		rowa = WarpShuffle(state[0].x, 0, 4) & 15;
		reduceDuplexRowt_8_v2h(prev,iterator,rowa, state, thread, threads);


		DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x] = state[0];
		DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x] = state[1];
		DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x] = state[2];
		DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x] = state[3];
	}
}

__global__ __launch_bounds__(64, 1)
void lyra2h_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *g_hash, uint32_t *resNonces)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint28 state[4];

	if (thread < threads)
	{
		state[0] = __ldg4(&((uint2x4*)DMatrix)[threads * 0 + thread]);
		state[1] = __ldg4(&((uint2x4*)DMatrix)[threads * 1 + thread]);
		state[2] = __ldg4(&((uint2x4*)DMatrix)[threads * 2 + thread]);
		state[3] = __ldg4(&((uint2x4*)DMatrix)[threads * 3 + thread]);

		for (int i = 0; i < 12; i++)
			round_lyra(state);
		uint32_t nonce = startNounce + thread;
		if (((uint64_t*)state)[3] <= ((uint64_t*)pTarget)[3]) {
			atomicMin(&resNonces[1], resNonces[0]);
			atomicMin(&resNonces[0], nonce);
		}
/*
		g_hash[thread + threads * 0] = state[0].x;
		g_hash[thread + threads * 1] = state[0].y;
		g_hash[thread + threads * 2] = state[0].z;
		g_hash[thread + threads * 3] = state[0].w;
*/
	} //thread
}


__global__ __launch_bounds__(64, 1)
void lyra2h_gpu_hash_32_1_sm5(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	const uint2x4 blake2b_IV[2] = {
		{ { 0xf3bcc908, 0x6a09e667 },{ 0x84caa73b, 0xbb67ae85 },{ 0xfe94f82b, 0x3c6ef372 },{ 0x5f1d36f1, 0xa54ff53a } },
		{ { 0xade682d1, 0x510e527f },{ 0x2b3e6c1f, 0x9b05688c },{ 0xfb41bd6b, 0x1f83d9ab },{ 0x137e2179, 0x5be0cd19 } }
	};
	const uint2x4 Mask[2] = {
		0x00000020UL, 0x00000000UL, 0x00000020UL, 0x00000000UL,
		0x00000020UL, 0x00000000UL, 0x00000010UL, 0x00000000UL,
		0x00000010UL, 0x00000000UL, 0x00000010UL, 0x00000000UL,
		0x00000080UL, 0x00000000UL, 0x00000000UL, 0x01000000UL
	};
	if (thread < threads)
	{
		uint2x4 state[4];

		((uint2*)state)[0] = __ldg(&g_hash[thread]);
		((uint2*)state)[1] = __ldg(&g_hash[thread + threads]);
		((uint2*)state)[2] = __ldg(&g_hash[thread + threads * 2]);
		((uint2*)state)[3] = __ldg(&g_hash[thread + threads * 3]);

		state[1] = state[0];
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i < 12; i++)
			round_lyra(state); //because 12 is not enough

		state[0] ^= Mask[0];
		state[1] ^= Mask[1];

		for (int i = 0; i < 12; i++)
			round_lyra(state); //because 12 is not enough


		((uint2x4*)DMatrix)[0 * threads + thread] = state[0];
		((uint2x4*)DMatrix)[1 * threads + thread] = state[1];
		((uint2x4*)DMatrix)[2 * threads + thread] = state[2];
		((uint2x4*)DMatrix)[3 * threads + thread] = state[3];
	}
}

__global__ __launch_bounds__(8, 1)
void lyra2h_gpu_hash_32_2_sm5(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.y * blockIdx.x + threadIdx.y);

	if (thread < threads)
	{
		uint2 state[4];

		state[0] = __ldg(&DMatrix[(0 * threads + thread)*blockDim.x + threadIdx.x]);
		state[1] = __ldg(&DMatrix[(1 * threads + thread)*blockDim.x + threadIdx.x]);
		state[2] = __ldg(&DMatrix[(2 * threads + thread)*blockDim.x + threadIdx.x]);
		state[3] = __ldg(&DMatrix[(3 * threads + thread)*blockDim.x + threadIdx.x]);

		reduceDuplexV5h(state, thread, threads);

		uint32_t rowa; // = WarpShuffle(state[0].x, 0, 4) & 7;
		uint32_t prev = 15;
		uint32_t iterator = 0;
		for (int tau = 0; tau<7; tau++) {
			for (uint32_t i = 0; i<16; i++) {
				rowa = WarpShuffle(state[0].x, 0, 4) & 15;
				reduceDuplexRowV50h(prev, rowa, iterator, state, thread, threads);
				prev = iterator;
				iterator = (iterator + 7) & 15;
			}
			for (uint32_t i = 0; i<16; i++) {
				rowa = WarpShuffle(state[0].x, 0, 4) & 15;
				reduceDuplexRowV50h(prev, rowa, iterator, state, thread, threads);
				prev = iterator;
				iterator = (iterator - 1) & 15;
			}
		}

		for (uint32_t i = 0; i<16; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 15;
			reduceDuplexRowV50h(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 7) & 15;
		}
		for (uint32_t i = 0; i<15; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 15;
			reduceDuplexRowV50h(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 15;
		}

		rowa = WarpShuffle(state[0].x, 0, 4) & 15;
		reduceDuplexRowV50_8_v2h(prev, iterator, rowa, state, thread, threads);



		DMatrix[(0 * threads + thread)*blockDim.x + threadIdx.x] = state[0];
		DMatrix[(1 * threads + thread)*blockDim.x + threadIdx.x] = state[1];
		DMatrix[(2 * threads + thread)*blockDim.x + threadIdx.x] = state[2];
		DMatrix[(3 * threads + thread)*blockDim.x + threadIdx.x] = state[3];
	}
}

__global__ __launch_bounds__(64, 1)
void lyra2h_gpu_hash_32_3_sm5(uint32_t threads, uint32_t startNounce, uint2 *g_hash, uint32_t *resNonces)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint2x4 state[4];

		state[0] = __ldg4(&((uint2x4*)DMatrix)[0 * threads + thread]);
		state[1] = __ldg4(&((uint2x4*)DMatrix)[1 * threads + thread]);
		state[2] = __ldg4(&((uint2x4*)DMatrix)[2 * threads + thread]);
		state[3] = __ldg4(&((uint2x4*)DMatrix)[3 * threads + thread]);

		for (int i = 0; i < 12; i++)
			round_lyra(state);


		uint32_t nonce = startNounce + thread;
		if (((uint64_t*)state)[3] <= ((uint64_t*)pTarget)[3]) {
			atomicMin(&resNonces[1], resNonces[0]);
			atomicMin(&resNonces[0], nonce);
		}

	}
}




__global__	__launch_bounds__(48, 1)
void lyra2h_gpu_hash_32_v3(uint32_t threads, uint32_t startNounce, uint2 *outputHash, uint32_t *resNonces)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	ulonglong4 state[4];


	const ulonglong4 blake2b_IV[2] = {
		{ 0x6a09e667f3bcc908,
		0xbb67ae8584caa73b,
		0x3c6ef372fe94f82b,
		0xa54ff53a5f1d36f1 },
		{ 0x510e527fade682d1,
		0x9b05688c2b3e6c1f,
		0x1f83d9abfb41bd6b,
		0x5be0cd19137e2179 } };

	const ulonglong4 Mask[2] = {
		{ 0x20,
		0x20,
		0x20,
		0x08 },
		{ 0x08,
		0x08,
		0x80,
		0x0100000000000000 }
	};

#if __CUDA_ARCH__ == 350 || __CUDA_ARCH__ == 370
	if (thread < threads)
#endif
	{

		((uint2*)state)[0] = __ldg(&outputHash[thread]);
		((uint2*)state)[1] = __ldg(&outputHash[thread + threads]);
		((uint2*)state)[2] = __ldg(&outputHash[thread + threads * 2]);
		((uint2*)state)[3] = __ldg(&outputHash[thread + threads * 3]);

		state[1] = state[0];

		state[2] = ((ulonglong4*)blake2b_IV)[0];
		state[3] = ((ulonglong4*)blake2b_IV)[1];

		for (int i = 0; i<12; i++)
			round_lyra_v35(state);  //because 12 is not enough

		state[0] ^= Mask[0];
		state[1] ^= Mask[1];

		for (int i = 0; i<12; i++)
			round_lyra_v35(state);  //because 12 is not enough


		uint32_t ps1 = (8 * memshift * 7 + 64 * memshift * thread);


		for (int i = 0; i < 8; i++)
		{
			uint32_t s1 = ps1 - 8 * memshift * i;
			for (int j = 0; j < 3; j++)
				((ulonglong4*)(DMatrix35))[j + s1] = (state)[j];

			round_lyra_v35(state);
		}


		reduceDuplexV3(state, thread);

		reduceDuplexRowSetupV3(1, 0, 2, state, thread);
		reduceDuplexRowSetupV3(2, 1, 3, state, thread);
		reduceDuplexRowSetupV3(3, 0, 4, state, thread);
		reduceDuplexRowSetupV3(4, 3, 5, state, thread);
		reduceDuplexRowSetupV3(5, 2, 6, state, thread);
		reduceDuplexRowSetupV3(6, 1, 7, state, thread);

		uint32_t rowa;
		uint32_t prev = 7;
		uint32_t iterator = 0;
		for (int j = 0; j<4; j++) {
			for (uint32_t i = 0; i<8; i++) {
				rowa = ((uint2*)state)[0].x & 7;
				reduceDuplexRowtV3(prev, rowa, iterator, state, thread);
				prev = iterator;
				iterator = (iterator + 3) & 7;
			}

			for (uint32_t i = 0; i<8; i++) {
				rowa = ((uint2*)state)[0].x & 7;
				reduceDuplexRowtV3(prev, rowa, iterator, state, thread);
				prev = iterator;
				iterator = (iterator - 1) & 7;
			}

		}

		uint32_t shift = (memshift * rowa + 64 * memshift * thread);

		for (int j = 0; j < 3; j++)
			state[j] ^= __ldg4(&((ulonglong4*)(DMatrix35))[j + shift]);

		for (int i = 0; i < 12; i++)
			round_lyra_v35(state);

		uint32_t nonce = startNounce + thread;
		if (((uint64_t*)state)[3] <= ((uint64_t*)pTarget)[3]) {
			atomicMin(&resNonces[1], resNonces[0]);
			atomicMin(&resNonces[0], nonce);
		}

	} //thread
}



__global__	__launch_bounds__(4 * 16, 1)
void lyra2h_gpu_hash_32_ws2(uint32_t threads, uint32_t startNounce, uint2 *outputHash, uint32_t *resNonces)
{
#if __CUDA_ARCH__ > 300 

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint32_t thread2 = thread >> 2;

	vectype state;
	vectype temp[16];


	const uint28 blake2b_IV[2] = {
		{ 0xf3bcc908, 0x6a09e667,
		0x84caa73b, 0xbb67ae85,
		0xfe94f82b, 0x3c6ef372,
		0x5f1d36f1, 0xa54ff53a },
		{ 0xade682d1, 0x510e527f,
		0x2b3e6c1f, 0x9b05688c,
		0xfb41bd6b, 0x1f83d9ab,
		0x137e2179, 0x5be0cd19 } };

	const uint28 padding[2] = {
		{ 0x20, 0x0,
		0x20, 0x0,
		0x20, 0x0,
		0x10, 0x0 },
		{ 0x10, 0x0,
		0x10, 0x0,
		0x80, 0x0,
		0x00, 0x01000000 } };

	//		state.x = __ldg(&((u64type*)outputHash)[4 * thread2 + (threadIdx.x % 4)]);

	state.x = __ldg(&((u64type*)outputHash)[thread2 + (threadIdx.x % 4) * threads]);
	state.y = state.x;

	state.z = ((u64type*)blake2b_IV)[threadIdx.x % 4];
	state.w = ((u64type*)blake2b_IV)[4 + threadIdx.x % 4];


	for (int i = 0; i<12; i++)
		round_lyra_v35_ws(state);

	state.x ^= ((u64type*)padding)[threadIdx.x % 4];
	state.y ^= ((u64type*)padding)[4 + threadIdx.x % 4];


	for (int i = 0; i<12; i++)
		round_lyra_v35_ws(state);

	uint32_t ps1 = (memshift * (Ncol - 1) + Nrow * Ncol * memshift * thread2);

	for (int i = 0; i < Ncol; i++)
	{
		uint32_t s1 = ps1 - memshift * i;
		((vectype*)(DMatrix35))[(threadIdx.x % 4) + s1] = state;
		temp[(15 - i)] = state;
		round_lyra_v35_ws(state);
	}

	reduceDuplex_ws2(state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass1(1, 0, 2, state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass2(2, 1, 3, state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass1(3, 0, 4, state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass2(4, 3, 5, state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass1(5, 2, 6, state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass2(6, 1, 7, state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass1(7, 0, 8, state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass2(8, 3, 9, state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass1(9, 6, 10, state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass2(10, 1, 11, state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass1(11, 4, 12, state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass2(12, 7, 13, state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass1(13, 2, 14, state, thread2, temp);
	reduceDuplexRowSetup_ws2_pass2(14, 5, 15, state, thread2, temp);

	uint32_t rowa;
	uint32_t prev = 15;
	uint32_t iterator = 0;
	for (int j = 0; j<8; j++) {
		for (int i = 0; i < Ncol; i++)
		{
			rowa = shuffle2t(state.x, 0, 4) & 15;
			reduceDuplexRow_ws2(prev, rowa, iterator, state, thread2, temp);
			prev = iterator;
			iterator = (iterator + 7) & 15;
		}
		for (int i = 0; i < Ncol; i++)
		{
			rowa = shuffle2t(state.x, 0, 4) & 15;
			if (i == 15 && j == 7)
				reduceDuplexRow_ws2_v2(prev, rowa, iterator, state, thread2, temp);
			else
				reduceDuplexRow_ws2(prev, rowa, iterator, state, thread2, temp);

			prev = iterator;
			iterator = (iterator - 1) & 15;
		}
	}

	uint32_t shift = (memshift * Ncol * rowa + Nrow * Ncol * memshift * thread2);
	vectype tmp2 = temp[15];//__ldg4t(&((vectype*)(DMatrix35))[(threadIdx.x % 4) + shift]);
	state.x ^= tmp2.x;
	state.y ^= tmp2.y;
	state.z ^= tmp2.z;

	for (int i = 0; i < 12; i++)
		round_lyra_v35_ws(state);

	//	((u64type*)outputHash)[4 * thread2 + (threadIdx.x % 4)] = state.x;

	if ((threadIdx.x % 4) == 3) {
		uint32_t nonce = startNounce + thread2;
		if (devectorize(state.x) <= ((uint64_t*)pTarget)[3]) {
			atomicMin(&resNonces[1], resNonces[0]);
			atomicMin(&resNonces[0], nonce);
		}
	}
#endif
}





__host__
void lyra2h_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix)
{

	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DMatrix, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
	cudaMalloc(&d_GNonces[thr_id], 2 * sizeof(uint32_t));
	cudaMallocHost(&h_GNonces[thr_id], 2 * sizeof(uint32_t));
}

__host__
void lyra2h_cpu_init_sm2(int thr_id, uint32_t threads)
{

	// just assign the device pointer allocated in main loop
	cudaMalloc(&d_GNonces[thr_id], 2 * sizeof(uint32_t));
	cudaMallocHost(&h_GNonces[thr_id], 2 * sizeof(uint32_t));
}
  

__host__
void lyra2h_cpu_init_sm35(int thr_id, uint32_t threads, uint64_t *d_matrix)
{

	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DMatrix35, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
	cudaMalloc(&d_GNonces[thr_id], 2 * sizeof(uint32_t));
	cudaMallocHost(&h_GNonces[thr_id], 2 * sizeof(uint32_t));
}


__host__
uint32_t lyra2h_getSecNonce(int thr_id, int num)
{
	uint32_t results[2];
	memset(results, 0xFF, sizeof(results));
	cudaMemcpy(results, d_GNonces[thr_id], sizeof(results), cudaMemcpyDeviceToHost);
	if (results[1] == results[0])
		return UINT32_MAX;
	return results[num];
}

__host__
void lyra2h_setTarget(const void *pTargetIn)
{
	cudaMemcpyToSymbol(pTarget, pTargetIn, 32, 0, cudaMemcpyHostToDevice);
}


__host__
uint32_t lyra2h_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, bool gtx750ti)
{
	uint32_t result = UINT32_MAX;
	cudaMemset(d_GNonces[thr_id], 0xff, 2 * sizeof(uint32_t));
	int dev_id = device_map[thr_id % MAX_GPUS];

	uint32_t tpb = TPB52;

	if (device_sm[dev_id] == 500) 
		tpb = TPB52;
	if (device_sm[dev_id] == 200) 
		tpb = TPB52;
	
	uint32_t tpb3 = 256;

	dim3 grid1((threads * 4 + tpb - 1) / tpb);
	dim3 block1(4, tpb >> 2);

	dim3 grid2((threads + 64 - 1) / 64);
	dim3 block2(64);

	dim3 grid3((threads + tpb3 - 1) / tpb3);
	dim3 block3(tpb3);
	
	if (device_sm[dev_id] >= 500)
	{
		lyra2h_gpu_hash_32_1 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash);

		lyra2h_gpu_hash_32_2 <<< grid1, block1, 768 *sizeof(uint2) * tpb >>> (threads, startNounce, d_hash);

		lyra2h_gpu_hash_32_3 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash, d_GNonces[thr_id]);
	}
	else if (device_sm[dev_id] == 350 || device_sm[dev_id] == 370)
	{

		uint32_t tpb35 = 16;
		dim3 grid35_ws(threads / tpb35);
		dim3 block35_ws(4 * tpb35);

		lyra2h_gpu_hash_32_ws2 << <grid35_ws, block35_ws >> > (threads, startNounce, (uint2*)d_hash, d_GNonces[thr_id]);

	}
	else 
		lyra2h_gpu_hash_32_sm2 <<< grid3, block3 >>> (threads, startNounce, d_hash, d_GNonces[thr_id]);



	// get first found nonce
	cudaMemcpy(h_GNonces[thr_id], d_GNonces[thr_id], 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	result = *h_GNonces[thr_id];

	return result;

}
