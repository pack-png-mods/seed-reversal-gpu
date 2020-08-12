// IDE indexing
#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__
#define __CUDACC__
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_cmath.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#include <device_functions.h>
#else
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <ctime>

using signed_seed_t = std::int64_t;
using uint = std::uint32_t;
using ulong = std::uint64_t;

#undef JRAND_DOUBLE

constexpr auto RANDOM_MULTIPLIER_LONG = 0x5DEECE66DULL;

#ifdef JRAND_DOUBLE
#define Random double
#define RANDOM_MULTIPLIER 0x5DEECE66Dp-48
#define RANDOM_ADDEND 0xBp-48
#define RANDOM_SCALE 0x1p-48
// should be signed with std::int32_t (to verify)
inline uint __host__ __device__ random_next(Random* random, int bits)
{
    *random = trunc((*random * RANDOM_MULTIPLIER + RANDOM_ADDEND) * RANDOM_SCALE);
    return (uint)((ulong)(*random / RANDOM_SCALE) >> (48 - bits));
}

#else

using Random = ulong;
constexpr auto RANDOM_MULTIPLIER = RANDOM_MULTIPLIER_LONG;
constexpr auto RANDOM_ADDEND = 0xBULL;
constexpr auto RANDOM_MASK = (1ULL << 48) - 1;
constexpr auto RANDOM_SCALE = 1;

#define FAST_NEXT_INT

// Random::next(bits)
__host__ __device__ inline uint random_next(Random* random, std::int32_t bits)
{
    *random = (*random * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
    return (uint)(*random >> (48 - bits));
}
#endif // ~JRAND_DOUBLE

// new Random(seed)

__host__ __device__ constexpr Random get_random(ulong seed)
{
    return ((seed ^ RANDOM_MULTIPLIER_LONG) & RANDOM_MASK);
}

__host__ __device__ constexpr Random get_random_unseeded(ulong state)
{
    return state * RANDOM_SCALE;
}

// Random::nextInt(bound)
__host__ __device__ inline uint random_next_int(Random* random, uint bound)
{
    std::int32_t r = random_next(random, 31);
    std::int32_t m = bound - 1;
    if ((bound & m) == 0) {
        // Could probably use __mul64hi here
        r = (uint)((bound * (ulong)r) >> 31);
    } else {
#ifdef FAST_NEXT_INT
        r %= bound;
#else
        for (std::int32_t u = r;
             u - (r = u % bound) + m < 0;
             u = random_next(random, 31))
            ;
#endif
    }
    return r;
}

__host__ __device__ inline signed_seed_t random_next_long(Random* random)
{
    return (((signed_seed_t)random_next(random, 32)) << 32) + (std::int32_t)random_next(random, 32);
}

inline void gpuAssert(cudaError_t code, const char* file = __FILE__, std::int32_t line = __LINE__)
{
    if (code != cudaSuccess) {
        std::fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
        std::exit(code);
    }
}

// advance

__host__ __device__ constexpr decltype(auto) advance(ulong& rand, signed_seed_t multiplier, signed_seed_t addend)
{
    return (rand = ((rand) * (multiplier) + (addend)) & RANDOM_MASK);
}
__host__ __device__ constexpr decltype(auto) advance_830(ulong& rand)
{
    return advance(rand, 0x859D39E832D9LL, 0xE3E2DF5E9196LL);
}
__host__ __device__ decltype(auto) advance_774(ulong& rand)
{
    return advance(rand, 0xF8D900133F9LL, 0x5738CAC2F85ELL);
}

__host__ __device__ decltype(auto) advance_387(ulong& rand)
{
    return advance(rand, 0x5FE2BCEF32B5LL, 0xB072B3BF0CBDLL);
}

__host__ __device__ decltype(auto) advance_16(ulong& rand)
{
    return advance(rand, 0x6DC260740241LL, 0xD0352014D90LL);
}

__host__ __device__ decltype(auto) advance_m1(ulong& rand)
{
    return advance(rand, 0xDFE05BCB1365LL, 0x615C0E462AA9LL);
}

__host__ __device__ decltype(auto) advance_m3759(ulong& rand)
{
    return advance(rand, 0x63A9985BE4ADLL, 0xA9AA8DA9BC9BLL);
}
constexpr auto TREE_X = 4;
constexpr auto TREE_Z = 3;
constexpr auto TREE_HEIGHT = 6;

constexpr auto OTHER_TREE_COUNT = 3;
__device__ inline std::int32_t getTreeHeight(std::int32_t x, std::int32_t z)
{
    if (x == TREE_X && z == TREE_Z)
        return TREE_HEIGHT;

    if (x == 1 && z == 13)
        return 5;

    if (x == 6 && z == 12)
        return 6;

    if (x == 14 && z == 7)
        return 5;

    return 0;
}

constexpr auto WATERFALL_X = 9;
constexpr auto WATERFALL_Y = 76;
constexpr auto WATERFALL_Z = 1;

constexpr auto MODULUS = 1LL << 48;
constexpr auto SQUARE_SIDE = MODULUS / 16;
constexpr auto X_TRANSLATE = 0;
constexpr auto Z_TRANSLATE = 11;
constexpr auto L00 = 7847617LL;
constexpr auto L01 = -18218081LL;
constexpr auto L10 = 4824621LL;
constexpr auto L11 = 24667315LL;
constexpr auto LI00 = 24667315.0 / 16;
constexpr auto LI01 = 18218081.0 / 16;
constexpr auto LI10 = -4824621.0 / 16;
constexpr auto LI11 = 7847617.0 / 16;

constexpr auto cuFloor(double x)
{
    return (signed_seed_t)x;
}

constexpr auto cuCeil(double x)
{
    return x == (signed_seed_t)x ? (signed_seed_t)x : cuFloor(x + 1);
}

// for a parallelogram ABCD https://media.discordapp.net/attachments/668607204009574411/671018577561649163/unknown.png
#define B_X LI00
#define B_Z LI10
constexpr auto C_X = LI00 + LI01;
constexpr auto C_Z = LI10 + LI11;
#define D_X LI01
#define D_Z LI11
constexpr auto LOWER_X = std::min<double>(std::min<double>(std::min<double>(0, B_X), C_X), D_X);
constexpr auto LOWER_Z = std::min<double>(std::min<double>(std::min<double>(0, B_Z), C_Z), D_Z);
constexpr auto UPPER_X = std::max<double>(std::max<double>(std::max<double>(0, B_X), C_X), D_X);
constexpr auto UPPER_Z = std::max<double>(std::max<double>(std::max<double>(0, B_Z), C_Z), D_Z);
constexpr auto ORIG_SIZE_X = (UPPER_X - LOWER_X + 1);
constexpr auto SIZE_X = cuCeil(ORIG_SIZE_X - D_X);
constexpr auto SIZE_Z = cuCeil(UPPER_Z - LOWER_Z + 1);
constexpr auto TOTAL_WORK_SIZE = (SIZE_X * SIZE_Z);

constexpr auto MAX_TREE_ATTEMPTS = 12;
constexpr auto MAX_TREE_SEARCH_BACK = 3 * MAX_TREE_ATTEMPTS - 3 + 16 * OTHER_TREE_COUNT;

__constant__ ulong search_back_multipliers[MAX_TREE_SEARCH_BACK + 1];
__constant__ ulong search_back_addends[MAX_TREE_SEARCH_BACK + 1];
static std::int32_t search_back_count;

constexpr auto WORK_UNIT_SIZE = 1LL << 23;
constexpr auto BLOCK_SIZE = 256;

__global__ void doPreWork(ulong offset, Random* starts, std::int32_t* num_starts)
{
    // lattice tree position
    ulong global_id = blockIdx.x * blockDim.x + threadIdx.x;

    signed_seed_t lattice_x = (signed_seed_t)((offset + global_id) % SIZE_X) + LOWER_X;
    signed_seed_t lattice_z = (signed_seed_t)((offset + global_id) / SIZE_X) + LOWER_Z;
    lattice_z += (B_X * lattice_z < B_Z * lattice_x) * SIZE_Z;
    if (D_X * lattice_z > D_Z * lattice_x) {
        lattice_x += B_X;
        lattice_z += B_Z;
    }
    lattice_x += (signed_seed_t)(TREE_X * LI00 + TREE_Z * LI01);
    lattice_z += (signed_seed_t)(TREE_X * LI10 + TREE_Z * LI11);

    Random rand = (Random)((lattice_x * L00 + lattice_z * L01 + X_TRANSLATE) % MODULUS);
    advance_m1(rand);

    Random tree_start = rand;
    advance_m1(tree_start);

    bool res = random_next(&rand, 4) == TREE_X;
    res &= random_next(&rand, 4) == TREE_Z;
    res &= random_next_int(&rand, 3) == (ulong)(TREE_HEIGHT - 4);

    if (res) {
        std::int32_t index = atomicAdd(num_starts, 1);
        starts[index] = tree_start;
    }
}

__global__ void doWork(std::int32_t* num_starts, Random* tree_starts, std::int32_t* num_seeds, ulong* seeds, std::int32_t gpu_search_back_count)
{
    for (std::int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < *num_starts; i += blockDim.x * gridDim.x) {
        Random tree_start = tree_starts[i];

        for (std::int32_t treeBackCalls = 0; treeBackCalls <= gpu_search_back_count; treeBackCalls++) {
            Random start = (tree_start * search_back_multipliers[treeBackCalls] + search_back_addends[treeBackCalls]) & RANDOM_MASK;
            Random rand = start;

            bool this_res = true;

            if (random_next_int(&rand, 10) == 0)
                continue;

            char generated_tree[16][2] = {}; // this will zero out generated_tree

            std::int32_t treesMatched = 0;
            bool any_population_matches = false;
            for (std::int32_t treeAttempt = 0; treeAttempt <= MAX_TREE_ATTEMPTS; treeAttempt++) {
                std::int32_t treeX = random_next(&rand, 4);
                std::int32_t treeZ = random_next(&rand, 4);
                std::int32_t wantedTreeHeight = getTreeHeight(treeX, treeZ);
                std::int32_t treeHeight = random_next_int(&rand, 3) + 4;

                char& boolpack = generated_tree[treeX][treeZ / 8];
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

                    for (std::int32_t i = 0; i < 50; i++) {
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

                std::int32_t index = atomicAdd(num_seeds, 1);
                seeds[index] = start_chunk_rand;
            }

            advance_m1(start);
        }
    }
}

struct GPU_Node {
    std::int32_t GPU;
    std::int32_t* num_seeds;
    ulong* seeds;
    std::int32_t* num_tree_starts;
    Random* tree_starts;
};

void setup_gpu_node(GPU_Node* node, std::int32_t gpu)
{
    gpuAssert(cudaSetDevice(gpu));
    node->GPU = gpu;
    gpuAssert(cudaMallocManaged(&node->num_seeds, sizeof(*node->num_seeds)));
    gpuAssert(cudaMallocManaged(&node->seeds, (1LL << 20))); // approx 1MB
    gpuAssert(cudaMallocManaged(&node->num_tree_starts, sizeof(*node->num_tree_starts)));
    gpuAssert(cudaMallocManaged(&node->tree_starts, (sizeof(Random) * WORK_UNIT_SIZE)));
}

void destroy_gpu_node(GPU_Node* node, std::int32_t gpu)
{
    gpuAssert(cudaFree(node->num_seeds));
    gpuAssert(cudaFree(node->seeds));
    gpuAssert(cudaFree(node->num_tree_starts));
    gpuAssert(cudaFree(node->tree_starts));
}

void calculate_search_backs(std::int32_t GPU_COUNT)
{
    bool allow_search_back[MAX_TREE_SEARCH_BACK + 1] = {};

    for (std::int32_t i = 0; i <= MAX_TREE_ATTEMPTS - OTHER_TREE_COUNT - 1; i++) {
        allow_search_back[i * 3] = true;
    }

    for (std::int32_t tree = 0; tree < OTHER_TREE_COUNT; tree++) {
        for (std::int32_t i = 0; i <= MAX_TREE_SEARCH_BACK - 19; i++) {
            if (allow_search_back[i])
                allow_search_back[i + 19] = true;
        }
    }

    search_back_count = 0;
    ulong multiplier = 1;
    ulong addend = 0;
    ulong multipliers[MAX_TREE_SEARCH_BACK + 1];
    ulong addends[MAX_TREE_SEARCH_BACK + 1];
    for (std::int32_t i = 0; i <= MAX_TREE_SEARCH_BACK; i++) {
        if (allow_search_back[i]) {
            std::int32_t index = search_back_count++;
            multipliers[index] = multiplier;
            addends[index] = addend;
        }
        multiplier = (multiplier * 0xDFE05BCB1365LL) & RANDOM_MASK;
        addend = (0xDFE05BCB1365LL * addend + 0x615C0E462AA9LL) & RANDOM_MASK;
    }

    for (std::int32_t gpu = 0; gpu < GPU_COUNT; gpu++) {
        gpuAssert(cudaSetDevice(gpu));
        gpuAssert(cudaMemcpyToSymbol(search_back_multipliers, &multipliers, search_back_count * sizeof(*multipliers)));
        gpuAssert(cudaMemcpyToSymbol(search_back_addends, &addends, search_back_count * sizeof(*addends)));
    }
}

int main(int argc, char* argv[])
{
    std::int32_t GPU_COUNT = 1;
    for (std::int32_t i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
            case 'g':
                if (std::isdigit(argv[i][2]))
                    GPU_COUNT = std::atoi(argv[i] + 2);
                break;
            default:
                std::printf("Error: Flag not recognized.");
                return -1;
            }
        } else {
            std::printf("Error: Please specify flag before argument.");
            return -1;
        }
    }
    GPU_Node* nodes = new GPU_Node[GPU_COUNT];
    std::printf("Searching %lld total seeds...\n", TOTAL_WORK_SIZE);

    calculate_search_backs(GPU_COUNT);

    std::FILE* out_file = fopen("chunk_seeds.txt", "w");

    for (std::int32_t i = 0; i < GPU_COUNT; i++) {
        setup_gpu_node(&nodes[i], i);
    }

    ulong count = 0;
    std::clock_t lastIteration = clock();
    std::clock_t startTime = clock();
    for (ulong offset = 0; offset < TOTAL_WORK_SIZE;) {

        for (std::int32_t gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++) {
            gpuAssert(cudaSetDevice(gpu_index));

            *nodes[gpu_index].num_tree_starts = 0;
            doPreWork<<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE>>>(offset, nodes[gpu_index].tree_starts, nodes[gpu_index].num_tree_starts);
            offset += WORK_UNIT_SIZE;
        }

        for (std::int32_t gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++) {
            gpuAssert(cudaSetDevice(gpu_index));
            gpuAssert(cudaDeviceSynchronize());
        }

        for (std::int32_t gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++) {
            gpuAssert(cudaSetDevice(gpu_index));

            *nodes[gpu_index].num_seeds = 0;
            doWork<<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE>>>(nodes[gpu_index].num_tree_starts, nodes[gpu_index].tree_starts, nodes[gpu_index].num_seeds, nodes[gpu_index].seeds, search_back_count);
        }

        for (std::int32_t gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++) {
            gpuAssert(cudaSetDevice(gpu_index));
            gpuAssert(cudaDeviceSynchronize());

            for (std::int32_t i = 0, e = *nodes[gpu_index].num_seeds; i < e; i++) {
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
        double estimatedTime = (double)(TOTAL_WORK_SIZE - numSearched) / (double)(WORK_UNIT_SIZE * GPU_COUNT) * iterationTime;
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
        std::printf("Searched: %lld seeds. Found: %lld matches. Uptime: %.1fs. Speed: %.2fm seeds/s. Completion: %.3f%%. ETA: %.1f%c.\n", numSearched, count, timeElapsed, speed, progress, estimatedTime, suffix);
    }
    std::fclose(out_file);

    //free memory
    for (std::int32_t i = 0; i < GPU_COUNT; i++) {
        destroy_gpu_node(&nodes[i], i);
    }
    delete[] nodes;
}