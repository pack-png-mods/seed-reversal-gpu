
// IDE indexing
#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__
#define __CUDACC__
#include <device_functions.h>
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_cmath.h>
#endif



#include <stdio.h>


#define signed_seed_t long long
#define uint unsigned int
#define ulong unsigned signed_seed_t

#undef JRAND_DOUBLE

#define RANDOM_MULTIPLIER_LONG 0x5DEECE66DUL

#ifdef JRAND_DOUBLE
#define Random double
#define RANDOM_MULTIPLIER 0x5DEECE66Dp-48
#define RANDOM_ADDEND 0xBp-48
#define RANDOM_SCALE 0x1p-48

inline uint random_next(Random *random, int bits) {
  *random = trunc((*random * RANDOM_MULTIPLIER + RANDOM_ADDEND) * RANDOM_SCALE);
  return (uint)((ulong)(*random / RANDOM_SCALE) >> (48 - bits));
}

#else

#define Random ulong
#define RANDOM_MULTIPLIER RANDOM_MULTIPLIER_LONG
#define RANDOM_ADDEND 0xBUL
#define RANDOM_MASK (1UL << 48) - 1
#define RANDOM_SCALE 1

#define FAST_NEXT_INT

// Random::next(bits)
__host__ __device__ inline uint random_next(Random *random, int bits) {
    *random = (*random * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
    return (uint)(*random >> (48 - bits));
}
#endif // ~JRAND_DOUBLE

// new Random(seed)
#define get_random(seed) ((Random)((seed ^ RANDOM_MULTIPLIER_LONG) & RANDOM_MASK))
#define get_random_unseeded(state) ((Random) ((state) * RANDOM_SCALE))

// Random::nextInt(bound)
__host__ __device__ inline uint random_next_int(Random *random, uint bound) {
    int r = random_next(random, 31);
    int m = bound - 1;
    if ((bound & m) == 0) {
        r = (uint)((bound * (ulong)r) >> 31);
    } else {
#ifdef FAST_NEXT_INT
        r %= bound;
#else
        for (int u = r;
             u - (r = u % bound) + m < 0;
             u = random_next(random, 31));
#endif
    }
    return r;
}

__host__ __device__ inline long random_next_long (Random *random) {
    return (((long)random_next(random, 32)) << 32) + random_next(random, 32);
}

// advance
#define advance_m1(rand) ((rand) = ((rand) * 0xDFE05BCB1365LL + 0x615C0E462AA9LL) & RANDOM_MASK)



#define TREE_X 0
#define TREE_Z 0
#define TREE_HEIGHT 3

#define OTHER_TREE_COUNT 1
__constant__ const int OTHER_TREE_XS[OTHER_TREE_COUNT] = { 1 };



#define MODULUS (1LL << 48)
#define X_TRANSLATE 0
#define Z_TRANSLATE 11
#define L00 7847617LL
#define L01 4824621LL
#define L10 (-18218081LL)
#define L11 24667315LL
#define LI00 (24667315.0 / 16)
#define LI01 (-4824621.0 / 16)
#define LI10 (18218081.0 / 16)
#define LI11 (7847617.0 / 16)

#define CONST_FLOOR(x) ((x) < (signed_seed_t) (x) ? (signed_seed_t) (x) - 1 : (signed_seed_t) (x))
#define CONST_CEIL(x) ((x) == (signed_seed_t) (x) ? (signed_seed_t) (x) : CONST_FLOOR((x) + 1))
#define CONST_LOWER(x, m, c) ((m) < 0 ? ((x) + 1 - (double) (c) / MODULUS) * (m) : ((x) - (double) (c) / MODULUS) * (m))
#define CONST_UPPER(x, m, c) ((m) < 0 ? ((x) - (double) (c) / MODULUS) * (m) : ((x) + 1 - (double) (c) / MODULUS) * (m))

#define LOWER_X CONST_FLOOR(CONST_LOWER(TREE_X, LI00, X_TRANSLATE) + CONST_LOWER(TREE_Z, LI01, Z_TRANSLATE))
#define LOWER_Z CONST_FLOOR(CONST_LOWER(TREE_X, LI10, X_TRANSLATE) + CONST_LOWER(TREE_Z, LI11, Z_TRANSLATE))
#define UPPER_X CONST_CEIL(CONST_UPPER(TREE_X, LI00, X_TRANSLATE) + CONST_UPPER(TREE_Z, LI01, Z_TRANSLATE))
#define UPPER_Z CONST_CEIL(CONST_UPPER(TREE_X, LI10, X_TRANSLATE) + CONST_UPPER(TREE_Z, LI11, Z_TRANSLATE))
#define SIZE_X (UPPER_X - LOWER_X + 1)
#define SIZE_Z (UPPER_Z - LOWER_Z + 1)
#define TOTAL_WORK_SIZE (SIZE_X * SIZE_Z)

#define WORK_UNIT_SIZE (1LL << 32)
#define BLOCK_SIZE 256



__global__ void map(ulong offset, bool* result) {
    // lattice tree position
    ulong global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset + global_id >= TOTAL_WORK_SIZE) return;

    signed_seed_t lattice_x = (signed_seed_t) ((offset + global_id) % SIZE_X) + LOWER_X;
    signed_seed_t lattice_z = (signed_seed_t) ((offset + global_id) / SIZE_X) + LOWER_Z;
    Random rand = (Random) ((lattice_x * L00 + lattice_z * L10 + X_TRANSLATE) % MODULUS);

    advance_m1(rand);
    Random start = rand;

    bool res = random_next(&rand, 4) == 0;
    res &= random_next(&rand, 4) == 0;
    res &= random_next_int(&rand, 3) == (ulong) (TREE_HEIGHT - 4);



    result[global_id] = res;

}

int main() {
    printf("%f\n", LI01);
    printf("[%lld, %lld, %lld, %lld]: %lld * %lld = %lld\n", LOWER_X, LOWER_Z, UPPER_X, UPPER_Z, SIZE_X, SIZE_Z, TOTAL_WORK_SIZE);


    bool* result;
    cudaMallocManaged(&result, WORK_UNIT_SIZE);


    ulong count = 0;
    for (ulong offset = 0; offset < TOTAL_WORK_SIZE; offset += WORK_UNIT_SIZE) {
        map <<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE>>> (offset, result);
        cudaDeviceSynchronize();

        for (ulong i = 0; i < WORK_UNIT_SIZE; i++) {
            if (result[i])
                count++;
        }
        printf("%lld\n", count);
    }

}