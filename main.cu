
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



#include <memory.h>
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
#define advance(rand, multiplier, addend) ((rand) = ((rand) * (multiplier) + (addend)) & RANDOM_MASK)
#define advance_830(rand) advance(rand, 0x859D39E832D9LL, 0xE3E2DF5E9196LL)
#define advance_774(rand) advance(rand, 0xF8D900133F9LL, 0x5738CAC2F85ELL)
#define advance_387(rand) advance(rand, 0x5FE2BCEF32B5LL, 0xB072B3BF0CBDLL)
#define advance_16(rand) advance(rand, 0x6DC260740241LL, 0xD0352014D90LL)
#define advance_m1(rand) advance(rand, 0xDFE05BCB1365LL, 0x615C0E462AA9LL)
#define advance_m3759(rand) advance(rand, 0x63A9985BE4ADLL, 0xA9AA8DA9BC9BLL)



#define TREE_X 4
#define TREE_Z 3
#define TREE_HEIGHT 6

#define OTHER_TREE_COUNT 3
__device__ inline int getTreeHeight(int x, int z) {
    if (x == TREE_X && z == TREE_Z)
        return TREE_HEIGHT;

    if (x == 1 && z == 13)
        return 5;

    if (x == 6 && z == 12)
        return 6;

    if (x == 14 && z == 7) {
        return 5;
    }

    return 0;
}

#define WATERFALL_X 9
#define WATERFALL_Y 76
#define WATERFALL_Z 1



#define MODULUS (1LL << 48)
#define X_TRANSLATE 0
#define Z_TRANSLATE 11
#define L00 7847617LL
#define L01 (-18218081LL)
#define L10 4824621LL
#define L11 24667315LL
#define LI00 (24667315.0 / 16)
#define LI01 (18218081.0 / 16)
#define LI10 (-4824621.0 / 16)
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

#define MAX_TREE_ATTEMPTS 12
#define MAX_TREE_SEARCH_BACK (3 * MAX_TREE_ATTEMPTS - 3 + 16 * OTHER_TREE_COUNT)

#define WORK_UNIT_SIZE (1LL << 20)
#define BLOCK_SIZE 256



__global__ void doWork(ulong offset, int* num_seeds, ulong* seeds) {
    // lattice tree position
    ulong global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset + global_id >= TOTAL_WORK_SIZE) return;

    signed_seed_t lattice_x = (signed_seed_t) ((offset + global_id) % SIZE_X) + LOWER_X;
    signed_seed_t lattice_z = (signed_seed_t) ((offset + global_id) / SIZE_X) + LOWER_Z;
    Random rand = (Random) ((lattice_x * L00 + lattice_z * L01 + X_TRANSLATE) % MODULUS);
    advance_m1(rand);
    Random start = rand;
    advance_m1(start);

    bool res = random_next(&rand, 4) == TREE_X;
    res &= random_next(&rand, 4) == TREE_Z;
    res &= random_next_int(&rand, 3) == (ulong) (TREE_HEIGHT - 4);


    for (int treeBackCalls = 0; treeBackCalls <= MAX_TREE_SEARCH_BACK; treeBackCalls++) {
        rand = start;

        bool this_res = res;
        this_res &= random_next_int(&rand, 10) != 0;

        bool generated_tree[16][16];
        memset(generated_tree, false, sizeof(generated_tree));

        int treesMatched = 0;
        bool any_population_matches = false;
        for (int treeAttempt = 0; treeAttempt <= MAX_TREE_ATTEMPTS; treeAttempt++) {
            int treeX = random_next(&rand, 4);
            int treeZ = random_next(&rand, 4);
            int wantedTreeHeight = getTreeHeight(treeX, treeZ);
            int treeHeight = random_next_int(&rand, 3) + 4;
            if (treeHeight == wantedTreeHeight && !generated_tree[treeX][treeZ]) {
                treesMatched++;
                generated_tree[treeX][treeZ] = true;
                advance_16(rand);
            }

            if (treesMatched == OTHER_TREE_COUNT + 1) {
                Random before_rest = rand;
                // yellow flowers
                advance_774(rand);
                // red flowers
                if (random_next(&rand, 1) == 0) {
                    advance_387(rand);
                }
                // brown mushroom
                if (random_next(&rand, 2) == 0) {
                    advance_387(rand);
                }
                // red mushroom
                if (random_next(&rand, 3) == 0) {
                    advance_387(rand);
                }
                // reeds
                advance_830(rand);
                // pumpkins
                if (random_next(&rand, 5) == 0) {
                    advance_387(rand);
                }

                for (int i = 0; i < 50; i++) {
                    bool waterfall_matches = random_next(&rand, 4) == WATERFALL_X;
                    waterfall_matches &= random_next_int(&rand, random_next_int(&rand, 120) + 8) == WATERFALL_Y;
                    waterfall_matches &= random_next(&rand, 4) == WATERFALL_Z;
                    any_population_matches |= waterfall_matches;
                }
                rand = before_rest;
            }
        }

        this_res &= any_population_matches;

        Random start_chunk_rand = start;
        advance_m3759(start_chunk_rand);
        if (this_res) {
            int index = atomicAdd(num_seeds, 1);
            seeds[index] = start_chunk_rand;
        }

        advance_m1(start);
    }

}

int main() {
    printf("Searching %lld total seeds...\n", TOTAL_WORK_SIZE);

    FILE* out_file = fopen("chunk_seeds.txt", "w");


    int* num_seeds;
    cudaMallocManaged(&num_seeds, sizeof(*num_seeds));

    ulong* seeds;
    cudaMallocManaged(&seeds, (1LL << 30)); // approx 1gb

    ulong count = 0;
    for (ulong offset = 0; offset < TOTAL_WORK_SIZE; offset += WORK_UNIT_SIZE) {
        *num_seeds = 0;

        doWork <<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE>>> (offset, num_seeds, seeds);
        cudaDeviceSynchronize();

        for (int i = 0, e = *num_seeds; i < e; i++) {
            fprintf(out_file, "%lld\n", seeds[i]);
        }
        fflush(out_file);

        count += *num_seeds;
        printf("Searched %lld seeds, found %lld matches \n", offset + WORK_UNIT_SIZE, count);
    }

    fclose(out_file);

}