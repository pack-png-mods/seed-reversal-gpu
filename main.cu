
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
#include <ctype.h>


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

inline uint __host__ __device__  random_next(Random *random, int bits) {
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
        // Could probably use __mul64hi here
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
#define advance_m1(rand) advance(rand, 0xDFE05BCB1365LL, 0x615C0E462AA9LL)
#define advance_m3759(rand) advance(rand, 0x63A9985BE4ADLL, 0xA9AA8DA9BC9BLL)



#define WATERFALL_X 11
#define WATERFALL_Y 76
#define WATERFALL_Z 10

#define TREE_X (WATERFALL_X - 5)
#define TREE_Z (WATERFALL_Z - 8)
#define TREE_HEIGHT 5

#define OTHER_TREE_COUNT 1
__device__ inline int getTreeHeight(int x, int z) {
    if (x == TREE_X && z == TREE_Z)
        return TREE_HEIGHT;

    if (x == WATERFALL_X - 3 && z == WATERFALL_Z + 3)
        return 5;

    return 0;
}



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
#define MAX_TREE_SEARCH_BACK (3 * MAX_TREE_ATTEMPTS - 3 + 16 * OTHER_TREE_COUNT)

__constant__ ulong search_back_multipliers[MAX_TREE_SEARCH_BACK + 1];
__constant__ ulong search_back_addends[MAX_TREE_SEARCH_BACK + 1];
int search_back_count;

#define WORK_UNIT_SIZE (1LL << 23)
#define BLOCK_SIZE 256

__global__ void doPreWork(ulong offset, Random* starts, int* num_starts) {
    // lattice tree position
    ulong global_id = blockIdx.x * blockDim.x + threadIdx.x;

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
    advance_m1(tree_start);

    bool res = random_next(&rand, 4) == TREE_X;
    res &= random_next(&rand, 4) == TREE_Z;
    res &= random_next_int(&rand, 3) == (ulong) (TREE_HEIGHT - 4);

    if(res) {
        int index = atomicAdd(num_starts, 1);
        starts[index] = tree_start;
    }
}

__global__ void doWork(int* num_starts, Random* tree_starts, int* num_seeds, ulong* seeds, int gpu_search_back_count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < *num_starts; i += blockDim.x * gridDim.x) {
        Random tree_start = tree_starts[i];

        for (int treeBackCalls = 0; treeBackCalls <= gpu_search_back_count; treeBackCalls++) {
            Random start = (tree_start * search_back_multipliers[treeBackCalls] + search_back_addends[treeBackCalls]) & RANDOM_MASK;
            Random rand = start;

            bool this_res = true;

            if(random_next_int(&rand, 10) == 0)
                continue;

            char generated_tree[16][2];
            memset(generated_tree, 0x00, sizeof(generated_tree));

            int treesMatched = 0;
            bool any_population_matches = false;
            for (int treeAttempt = 0; treeAttempt <= MAX_TREE_ATTEMPTS; treeAttempt++) {
                int treeX = random_next(&rand, 4);
                int treeZ = random_next(&rand, 4);
                int wantedTreeHeight = getTreeHeight(treeX, treeZ);
                int treeHeight = random_next_int(&rand, 3) + 4;

                char& boolpack = generated_tree[treeX][treeZ / 2];
                const char mask = 1 << (treeZ % 8);

                if (treeHeight == wantedTreeHeight && !(boolpack & mask)) {
                    treesMatched++;
                    boolpack |= mask;
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

            if (this_res) {
                Random start_chunk_rand = start;
                advance_m3759(start_chunk_rand);

                int index = atomicAdd(num_seeds, 1);
                seeds[index] = start_chunk_rand;
            }

            advance_m1(start);
        }
    }
}

struct GPU_Node {
    int GPU;
    int* num_seeds;
    ulong* seeds;
    int* num_tree_starts;
    Random* tree_starts;
};

void setup_gpu_node(GPU_Node* node, int gpu) {
    CHECK_GPU_ERR(cudaSetDevice(gpu));
    node->GPU = gpu;
    CHECK_GPU_ERR(cudaMallocManaged(&node->num_seeds, sizeof(*node->num_seeds)));
    CHECK_GPU_ERR(cudaMallocManaged(&node->seeds, (1LL << 20))); // approx 1MB
    CHECK_GPU_ERR(cudaMallocManaged(&node->num_tree_starts, sizeof(*node->num_tree_starts)));
    CHECK_GPU_ERR(cudaMallocManaged(&node->tree_starts, (sizeof(Random)*WORK_UNIT_SIZE)));
}


void calculate_search_backs(int GPU_COUNT) {
    bool allow_search_back[MAX_TREE_SEARCH_BACK + 1];
    memset(allow_search_back, false, sizeof(allow_search_back));

    for (int i = 0; i <= MAX_TREE_ATTEMPTS - OTHER_TREE_COUNT - 1; i++) {
        allow_search_back[i * 3] = true;
    }

    for (int tree = 0; tree < OTHER_TREE_COUNT; tree++) {
        for (int i = 0; i <= MAX_TREE_SEARCH_BACK - 19; i++) {
            if (allow_search_back[i])
                allow_search_back[i + 19] = true;
        }
    }

    search_back_count = 0;
    ulong multiplier = 1;
    ulong addend = 0;
    ulong multipliers[MAX_TREE_SEARCH_BACK + 1];
    ulong addends[MAX_TREE_SEARCH_BACK + 1];
    for (int i = 0; i <= MAX_TREE_SEARCH_BACK; i++) {
        if (allow_search_back[i]) {
            int index = search_back_count++;
            multipliers[index] = multiplier;
            addends[index] = addend;
        }
        multiplier = (multiplier * 0xDFE05BCB1365LL) & RANDOM_MASK;
        addend = (0xDFE05BCB1365LL * addend + 0x615C0E462AA9LL) & RANDOM_MASK;
    }

    for (int gpu = 0; gpu < GPU_COUNT; gpu++) {
        CHECK_GPU_ERR(cudaSetDevice(gpu));
        CHECK_GPU_ERR(cudaMemcpyToSymbol(search_back_multipliers, &multipliers, search_back_count * sizeof(*multipliers)));
        CHECK_GPU_ERR(cudaMemcpyToSymbol(search_back_addends, &addends, search_back_count * sizeof(*addends)));
    }
}


#undef int
int main(int argc, char *argv[]) {
#define int int32_t
    int GPU_COUNT = 1;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            switch(argv[i][1]) {
                case 'g':
                    if(isdigit(argv[i][2])) GPU_COUNT = atoi(argv[i] + 2);
                    break;
                default:
                    printf("Error: Flag not recognized.");
                    return -1;
            }
        } else {
            printf("Error: Please specify flag before argument.");
            return -1;
        }
    }
    GPU_Node *nodes = (GPU_Node*)malloc(sizeof(GPU_Node) * GPU_COUNT);
    printf("Searching %lld total seeds...\n", TOTAL_WORK_SIZE);

    calculate_search_backs(GPU_COUNT);

    FILE* out_file = fopen("chunk_seeds.txt", "w");

    for(int i = 0; i < GPU_COUNT; i++) {
        setup_gpu_node(&nodes[i],i);
    }


    ulong count = 0;
    clock_t lastIteration = clock();
    clock_t startTime = clock();
    for (ulong offset = 0; offset < TOTAL_WORK_SIZE;) {

        for(int gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++) {
            CHECK_GPU_ERR(cudaSetDevice(gpu_index));

            *nodes[gpu_index].num_tree_starts = 0;
            doPreWork <<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE>>> (offset, nodes[gpu_index].tree_starts, nodes[gpu_index].num_tree_starts);
            offset += WORK_UNIT_SIZE;
        }

        for(int gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++) {
            CHECK_GPU_ERR(cudaSetDevice(gpu_index));
            CHECK_GPU_ERR(cudaDeviceSynchronize());
        }

        for(int gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++) {
            CHECK_GPU_ERR(cudaSetDevice(gpu_index));

            *nodes[gpu_index].num_seeds = 0;
            doWork <<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE>>> (nodes[gpu_index].num_tree_starts, nodes[gpu_index].tree_starts, nodes[gpu_index].num_seeds, nodes[gpu_index].seeds, search_back_count);
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

        double iterationTime = (double)(clock() - lastIteration) / CLOCKS_PER_SEC;
        double timeElapsed = (double)(clock() - startTime) / CLOCKS_PER_SEC;
        lastIteration = clock();
        ulong numSearched = offset + WORK_UNIT_SIZE * GPU_COUNT;
        double speed = (double)WORK_UNIT_SIZE * GPU_COUNT / (double)iterationTime / 1000000.0;
        double progress = (double)numSearched / (double)TOTAL_WORK_SIZE * 100.0;
        double estimatedTime = (double)(TOTAL_WORK_SIZE - numSearched) / (double) (WORK_UNIT_SIZE * GPU_COUNT) * iterationTime;
        char suffix = 's';
        if (estimatedTime >= 3600) {
            suffix = 'h';
            estimatedTime /= 3600.0;
        } else if (estimatedTime >= 60) {
            suffix = 'm';
            estimatedTime /= 60.0;
        }
        if (progress >= 100.0) {
            estimatedTime = 0.0;
            suffix = 's';
        }
        printf("Searched: %lld seeds. Found: %lld matches. Uptime: %.1fs. Speed: %.2fm seeds/s. Completion: %.3f%%. ETA: %.1f%c.\n", numSearched, count, timeElapsed, speed, progress, estimatedTime, suffix);

    }

    fclose(out_file);

}
