/*
 * cuBLAS FC-style GEMM benchmark (same math as one Linear layer without bias):
 *   Mode 1 — Baseline: cublasSgemm (FP32 on CUDA cores, no TF32 Tensor Ops).
 *   Mode 2 — Tensor Cores: cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32 (FP32 I/O, TF32 math).
 *
 * Column-major GEMM: C(m x n) = A(m x k) * B(k x n), matching cuBLAS layout.
 * CLI mirrors the Python script for comparable (B,K,N) triples.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
                         cudaGetErrorString(err__));                                     \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

#define CUBLAS_CHECK(call)                                                               \
    do {                                                                                 \
        cublasStatus_t st__ = (call);                                                    \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                            \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)st__); \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

static void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s --warmup N --iters M [--csv path] [--sizes B,K,N ...]\n"
                 "  If --sizes omitted, runs a default sweep.\n"
                 "  Produces rows for cublas_sgemm and cublas_gemm_ex_tf32.\n",
                 argv0);
}

static std::vector<std::tuple<int, int, int>> default_sweep() {
    std::vector<std::tuple<int, int, int>> out;
    const int batches[] = {64, 128, 256};
    const int dims[] = {256, 512, 1024, 2048, 4096};
    for (int b : batches) {
        for (int d : dims) {
            out.emplace_back(b, d, d);
        }
    }
    return out;
}

static std::vector<std::tuple<int, int, int>> parse_sizes(int argc, char** argv) {
    std::vector<std::tuple<int, int, int>> triples;
    bool in = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--sizes") == 0) {
            in = true;
            continue;
        }
        if (argv[i][0] == '-' && argv[i][1] == '-') {
            in = false;
            continue;
        }
        if (in) {
            int B = 0, K = 0, N = 0;
            if (std::sscanf(argv[i], "%d,%d,%d", &B, &K, &N) == 3) {
                triples.emplace_back(B, K, N);
            } else {
                std::fprintf(stderr, "Bad --sizes entry: %s (want B,K,N)\n", argv[i]);
                std::exit(EXIT_FAILURE);
            }
        }
    }
    if (triples.empty()) {
        return default_sweep();
    }
    return triples;
}

static int parse_int_flag(int argc, char** argv, const char* flag, int defv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) {
            return std::atoi(argv[i + 1]);
        }
    }
    return defv;
}

static std::string parse_string_flag(int argc, char** argv, const char* flag, const std::string& defv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) {
            return std::string(argv[i + 1]);
        }
    }
    return defv;
}

static inline long long gemm_flops(int m, int k, int n) {
    return 2LL * m * n * k;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const int warmup = parse_int_flag(argc, argv, "--warmup", 20);
    const int iters = parse_int_flag(argc, argv, "--iters", 100);
    const std::string csv_path = parse_string_flag(argc, argv, "--csv", "results_cublas_fc.csv");

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU: %s  SM %d.%d  CC major supports TF32 Tensor Ops on Ampere+ (SM80+)\n",
                prop.name, prop.major, prop.minor);

    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));

    auto triples = parse_sizes(argc, argv);

    std::ofstream csv;
    csv.open(csv_path, std::ios::out | std::ios::trunc);
    if (!csv) {
        std::fprintf(stderr, "Cannot open CSV: %s\n", csv_path.c_str());
        return EXIT_FAILURE;
    }
    csv << "mode,batch,in_features,out_features,avg_latency_ms,gemm_gflops\n";

    cudaEvent_t ev0{}, ev1{};
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // Use std::get (not structured bindings) for nvcc compatibility on some toolkits (e.g. Colab).
    for (const auto& t : triples) {
        const int B = std::get<0>(t);
        const int K = std::get<1>(t);
        const int N = std::get<2>(t);
        const int m = B;
        const int k = K;
        const int n = N;

        const size_t bytesA = sizeof(float) * static_cast<size_t>(m) * k;
        const size_t bytesB = sizeof(float) * static_cast<size_t>(k) * n;
        const size_t bytesC = sizeof(float) * static_cast<size_t>(m) * n;

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, bytesA));
        CUDA_CHECK(cudaMalloc(&dB, bytesB));
        CUDA_CHECK(cudaMalloc(&dC, bytesC));

        // Deterministic non-zero fill (host then copy) — not performance-critical
        std::vector<float> hA(m * k, 1.0f);
        std::vector<float> hB(k * n, 1.0f);
        std::vector<float> hC(m * n, 0.0f);
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dC, hC.data(), bytesC, cudaMemcpyHostToDevice));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Column-major leading dimensions
        const int lda = m;
        const int ldb = k;
        const int ldc = m;

        // --- cublasSgemm: FP32 GEMM (CUDA cores) ---
        for (int i = 0; i < warmup; ++i) {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, lda, dB, ldb,
                                    &beta, dC, ldc));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; ++i) {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, lda, dB, ldb,
                                    &beta, dC, ldc));
        }
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms_total = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_total, ev0, ev1));
        const float ms_sgemm = ms_total / static_cast<float>(iters);
        const double gflops_sgemm =
            static_cast<double>(gemm_flops(m, k, n)) / (static_cast<double>(ms_sgemm) / 1000.0) / 1e9;

        std::printf("cublas_sgemm               B=%5d K=%5d N=%5d  lat_ms=%.4f  GEMM_GFLOPS=%.2f\n", B, K, N,
                    ms_sgemm, gflops_sgemm);
        csv << "cublas_sgemm," << B << "," << K << "," << N << "," << ms_sgemm << "," << gflops_sgemm << "\n";

        // --- cublasGemmEx: TF32 Tensor Ops, FP32 I/O (CUDA 11+ cuBLAS) ---
        for (int i = 0; i < warmup; ++i) {
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, CUDA_R_32F, lda,
                                      dB, CUDA_R_32F, ldb, &beta, dC, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_FAST_TF32,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; ++i) {
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, CUDA_R_32F, lda,
                                      dB, CUDA_R_32F, ldb, &beta, dC, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_FAST_TF32,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventElapsedTime(&ms_total, ev0, ev1));
        const float ms_gemmex = ms_total / static_cast<float>(iters);
        const double gflops_gemmex =
            static_cast<double>(gemm_flops(m, k, n)) / (static_cast<double>(ms_gemmex) / 1000.0) / 1e9;

        std::printf("cublas_gemm_ex_tf32        B=%5d K=%5d N=%5d  lat_ms=%.4f  GEMM_GFLOPS=%.2f\n", B, K, N,
                    ms_gemmex, gflops_gemmex);
        csv << "cublas_gemm_ex_tf32," << B << "," << K << "," << N << "," << ms_gemmex << "," << gflops_gemmex
            << "\n";

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUBLAS_CHECK(cublasDestroy(handle));
    csv.close();
    std::printf("Wrote CSV: %s\n", csv_path.c_str());
    return EXIT_SUCCESS;
}
