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



#include <stdint.h>
#include <memory.h>
#include <stdio.h>
#include <time.h>


#define signed_seed_t int64_t
#define uint uint32_t
#define ulong uint64_t
// let's be EVIL (and make sure all includes come before this)
#define int int32_t

#undef JRAND_DOUBLE

#define RANDOM_MULTIPLIER_LONG 0x5DEECE66DULL

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
#define RANDOM_ADDEND 0xBULL
#define RANDOM_MASK (1ULL << 48) - 1
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

__host__ __device__ inline int64_t random_next_long (Random *random) {
    return (((int64_t)random_next(random, 32)) << 32) + random_next(random, 32);
}

#define CHECK_GPU_ERR(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
        exit(code);
    }
}

// advance
#define advance(rand, multiplier, addend) ((rand) = ((rand) * (multiplier) + (addend)) & RANDOM_MASK)
#define advance_830(rand) advance(rand, 0x859D39E832D9LL, 0xE3E2DF5E9196LL)
#define advance_774(rand) advance(rand, 0xF8D900133F9LL, 0x5738CAC2F85ELL)
#define advance_387(rand) advance(rand, 0x5FE2BCEF32B5LL, 0xB072B3BF0CBDLL)
#define advance_16(rand) advance(rand, 0x6DC260740241LL, 0xD0352014D90LL)
#define advance_2(rand) advance(rand, 0xBB20B4600A69LL, 0x40942DE6BALL)
#define advance_m1(rand) advance(rand, 0xDFE05BCB1365LL, 0x615C0E462AA9LL)
#define advance_m3759(rand) advance(rand, 0x63A9985BE4ADLL, 0xA9AA8DA9BC9BLL)



#define TREE_X 4
#define TREE_Z 3
#define TREE_HEIGHT 6

#define OTHER_TREE_COUNT 3
__device__ inline int getOtherTreeFlag(int x, int z, int height) {
    if (x == 1 && z == 13 && height == 5)
        return 1;

    if (x == 6 && z == 12 && height == 6)
        return 2;

    if (x == 14 && z == 7 && height == 5)
        return 4;

    return 0;
}

#define WATERFALL_X 9
#define WATERFALL_Y 76
#define WATERFALL_Z 1



#define MODULUS (1LL << 48)
#define SQUARE_SIDE (MODULUS / 16)
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

#define CONST_MIN(a, b) ((a) < (b) ? (a) : (b))
#define CONST_MIN4(a, b, c, d) CONST_MIN(CONST_MIN(a, b), CONST_MIN(c, d))
#define CONST_MAX(a, b) ((a) > (b) ? (a) : (b))
#define CONST_MAX4(a, b, c, d) CONST_MAX(CONST_MAX(a, b), CONST_MAX(c, d))
#define CONST_FLOOR(x) ((x) < (signed_seed_t) (x) ? (signed_seed_t) (x) - 1 : (signed_seed_t) (x))
#define CONST_CEIL(x) ((x) == (signed_seed_t) (x) ? (signed_seed_t) (x) : CONST_FLOOR((x) + 1))
#define CONST_LOWER(x, m, c) ((m) < 0 ? ((x) + 1 - (double) (c) / MODULUS) * (m) : ((x) - (double) (c) / MODULUS) * (m))
#define CONST_UPPER(x, m, c) ((m) < 0 ? ((x) - (double) (c) / MODULUS) * (m) : ((x) + 1 - (double) (c) / MODULUS) * (m))

// for a parallelogram ABCD https://media.discordapp.net/attachments/668607204009574411/671018577561649163/unknown.png
#define B_X LI00
#define B_Z LI10
#define C_X (LI00 + LI01)
#define C_Z (LI10 + LI11)
#define D_X LI01
#define D_Z LI11
#define LOWER_X CONST_MIN4(0, B_X, C_X, D_X)
#define LOWER_Z CONST_MIN4(0, B_Z, C_Z, D_Z)
#define UPPER_X CONST_MAX4(0, B_X, C_X, D_X)
#define UPPER_Z CONST_MAX4(0, B_Z, C_Z, D_Z)
#define ORIG_SIZE_X (UPPER_X - LOWER_X + 1)
#define SIZE_X CONST_CEIL(ORIG_SIZE_X - D_X)
#define SIZE_Z CONST_CEIL(UPPER_Z - LOWER_Z + 1)
#define TOTAL_WORK_SIZE (SIZE_X * SIZE_Z)

#define MAX_TREE_ATTEMPTS 12
#define MAX_TREE_SEARCH_COUNT ((3 * MAX_TREE_ATTEMPTS - 3 + 16 * OTHER_TREE_COUNT) * 2)
#define MAX_WATERFALL_SERACH_COUNT (387 * 5 + 4 * 50 + MAX_TREE_SEARCH_COUNT)

__constant__ ulong tree_search_multipliers[MAX_TREE_SEARCH_COUNT];
__constant__ ulong tree_search_addends[MAX_TREE_SEARCH_COUNT];
int tree_search_count;
__constant__ ulong waterfall_search_multipliers[MAX_WATERFALL_SERACH_COUNT];
__constant__ ulong waterfall_search_addends[MAX_WATERFALL_SERACH_COUNT];
int waterfall_search_count;

#define WORK_UNIT_SIZE (1LL << 23)
#define BLOCK_SIZE 256



__global__ void doWork(ulong offset, int* num_seeds, ulong* seeds, int gpu_tree_search_count, int gpu_waterfall_search_count) {
    // lattice tree position
    ulong global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset + global_id >= TOTAL_WORK_SIZE) return;

    signed_seed_t lattice_x = (signed_seed_t) ((offset + global_id) % SIZE_X) + LOWER_X;
    signed_seed_t lattice_z = (signed_seed_t) ((offset + global_id) / SIZE_X) + LOWER_Z;
    lattice_z += (B_X * lattice_z < B_Z * lattice_x) * SIZE_Z;
    if (D_X * lattice_z > D_Z * lattice_x) {
        lattice_x += B_X;
        lattice_z += B_Z;
    }
    lattice_x += (signed_seed_t) (TREE_X * LI00 + TREE_Z * LI01);
    lattice_z += (signed_seed_t) (TREE_X * LI10 + TREE_Z * LI11);
    Random rand = (Random) ((lattice_x * L00 + lattice_z * L01 + X_TRANSLATE) % MODULUS);
    advance_m1(rand);
    Random tree_start = rand;

    advance_2(rand); // x, z hopefully done by lattice
    bool valid = random_next_int(&rand, 3) + 4 == TREE_HEIGHT;

    int other_tree_flags = 0;

    for (int i = 0; i < gpu_tree_search_count; i++) {
        rand = (tree_start * tree_search_multipliers[i] + tree_search_addends[i]) & RANDOM_MASK;
        int x = random_next(&rand, 4);
        int z = random_next(&rand, 4);
        int height = random_next_int(&rand, 3) + 4;
        other_tree_flags |= getOtherTreeFlag(x, z, height);
    }

    valid &= other_tree_flags == (1 << OTHER_TREE_COUNT) - 1;

    bool any_waterfall_matches = false;
    for (int i = 0; i < gpu_waterfall_search_count; i++) {
        rand = (tree_start * waterfall_search_multipliers[i] + waterfall_search_addends[i]) & RANDOM_MASK;
        bool this_waterfall_matches = random_next(&rand, 4) == WATERFALL_X;
        this_waterfall_matches &= random_next_int(&rand, random_next_int(&rand, 120) + 8) == WATERFALL_Y;
        this_waterfall_matches &= random_next(&rand, 4) == WATERFALL_Z;
        any_waterfall_matches |= this_waterfall_matches;
    }

    valid &= any_waterfall_matches;

    if (valid) {
        int index = atomicAdd(num_seeds, 1);
        seeds[index] = tree_start;
    }
}

#define GPU_COUNT 1



struct GPU_Node {
    int GPU;
    int* num_seeds;
    ulong* seeds;
};
GPU_Node nodes[GPU_COUNT];

void setup_gpu_node(GPU_Node* node, int gpu) {
    CHECK_GPU_ERR(cudaSetDevice(gpu));
    node->GPU = gpu;
    CHECK_GPU_ERR(cudaMallocManaged(&node->num_seeds, sizeof(*node->num_seeds)));
    CHECK_GPU_ERR(cudaMallocManaged(&node->seeds, (1LL << 10))); // approx 1kb
}


void calculate_searches() {
    bool allow_tree_search[MAX_TREE_SEARCH_COUNT + 1];
    memset(allow_tree_search, false, sizeof(allow_tree_search));

    for (int i = 0; i <= MAX_TREE_ATTEMPTS - OTHER_TREE_COUNT - 1; i++) {
        allow_tree_search[i * 3] = true;
    }

    for (int tree = 0; tree < OTHER_TREE_COUNT; tree++) {
        for (int i = 0; i <= MAX_TREE_SEARCH_COUNT - 19; i++) {
            if (allow_tree_search[i])
                allow_tree_search[i + 19] = true;
        }
    }

    tree_search_count = 0;
    ulong multiplier = 1;
    ulong addend = 0;
    ulong multipliers[MAX_WATERFALL_SERACH_COUNT + 1];
    ulong addends[MAX_WATERFALL_SERACH_COUNT + 1];

    // backwards
    for (int i = 0; i <= MAX_TREE_SEARCH_COUNT; i++) {
        if (allow_tree_search[i]) {
            int index = tree_search_count++;
            multipliers[index] = multiplier;
            addends[index] = addend;
        }
        multiplier = (multiplier * 0xDFE05BCB1365LL) & RANDOM_MASK;
        addend = (0xDFE05BCB1365LL * addend + 0x615C0E462AA9LL) & RANDOM_MASK;
    }

    // forwards
    multiplier = 0x11117495BF5LL;
    addend = 0x409AA63C700DLL;
    for (int i = 0; i <= MAX_TREE_SEARCH_COUNT; i++) {
        if (allow_tree_search[i]) {
            int index = tree_search_count++;
            multipliers[index] = multiplier;
            addends[index] = addend;
        }
        multiplier = (multiplier * 0x5DEECE66DLL) & RANDOM_MASK;
        addend = (0x5DEECE66DLL * addend + 0xBLL) & RANDOM_MASK;
    }

    for (int gpu = 0; gpu < GPU_COUNT; gpu++) {
        CHECK_GPU_ERR(cudaSetDevice(gpu));
        CHECK_GPU_ERR(cudaMemcpyToSymbol(tree_search_multipliers, &multipliers, tree_search_count * sizeof(*multipliers)));
        CHECK_GPU_ERR(cudaMemcpyToSymbol(tree_search_addends, &addends, tree_search_count * sizeof(*addends)));
    }


    bool allow_waterfall_search[MAX_WATERFALL_SERACH_COUNT + 1];
    memset(allow_waterfall_search, false, sizeof(allow_waterfall_search));

    for (int tree = 0; tree <= MAX_TREE_SEARCH_COUNT; tree++) {
        if (allow_tree_search[tree]) {
            for (int waterfall_alignment = 0; waterfall_alignment <= 4; waterfall_alignment++) {
                for (int waterfall = 0; waterfall < 50; waterfall++) {
                    allow_waterfall_search[tree + 387 * waterfall_alignment + 4 * waterfall] = true;
                }
            }
        }
    }

    waterfall_search_count = 0;
    multiplier = 0x26FD89F9DA95LL;
    addend = 0x905AB48D4435LL;
    for (int i = 0; i <= MAX_WATERFALL_SERACH_COUNT; i++) {
        if (allow_waterfall_search[i]) {
            int index = waterfall_search_count++;
            multipliers[index] = multiplier;
            addends[index] = addend;
        }
        multiplier = (multiplier * 0x5DEECE66DLL) & RANDOM_MASK;
        addend = (0x5DEECE66DLL * addend + 0xBLL) & RANDOM_MASK;
    }

    for (int gpu = 0; gpu < GPU_COUNT; gpu++) {
        CHECK_GPU_ERR(cudaSetDevice(gpu));
        CHECK_GPU_ERR(cudaMemcpyToSymbol(waterfall_search_multipliers, &multipliers, waterfall_search_count * sizeof(*multipliers)));
        CHECK_GPU_ERR(cudaMemcpyToSymbol(waterfall_search_addends, &addends, waterfall_search_count * sizeof(*addends)));
    }
}


#undef int
int main() {
#define int int32_t
    printf("Searching %lld total seeds...\n", TOTAL_WORK_SIZE);

    calculate_searches();

    FILE* out_file = fopen("chunk_seeds.txt", "w");

    for(int i = 0; i < GPU_COUNT; i++) {
        setup_gpu_node(&nodes[i],i);
    }

    
    ulong count = 0;
    clock_t startTime = clock();
    for (ulong offset = 0; offset < TOTAL_WORK_SIZE;) {
        
        for(int gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++) {
            CHECK_GPU_ERR(cudaSetDevice(gpu_index));
            *nodes[gpu_index].num_seeds = 0;
            doWork <<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE>>> (offset, nodes[gpu_index].num_seeds, nodes[gpu_index].seeds, tree_search_count, waterfall_search_count);
            offset += WORK_UNIT_SIZE;
        }
        
        for(int gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++) {
            CHECK_GPU_ERR(cudaSetDevice(gpu_index));
            CHECK_GPU_ERR(cudaDeviceSynchronize());
            
            for (int i = 0, e = *nodes[gpu_index].num_seeds; i < e; i++) {
                fprintf(out_file, "%lld\n", nodes[gpu_index].seeds[i]);
            }
            fflush(out_file);
            count += *nodes[gpu_index].num_seeds;
        }
        
        double timeElapsed = (double)(clock() - startTime);
        timeElapsed /= CLOCKS_PER_SEC;
        ulong numSearched = offset + WORK_UNIT_SIZE;
        double speed = (double)numSearched / (double)timeElapsed / 1000000.0;
        double progress = (double)numSearched / (double)TOTAL_WORK_SIZE * 100.0;
        printf("Searched %lld seeds, found %lld matches. Time elapsed: %.1fs. Speed: %.2fm seeds/s. Completion: %.3f%%\n", numSearched, count, timeElapsed, speed, progress);

    }

    fclose(out_file);

}
