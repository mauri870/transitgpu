// The accumulation of the box search in the local memory of the device (LDS).
//
// A block takes `PB` points of one period, folds them into phase bins, and for each flux adds the quantized flux of its
// points into a private table of 64-bit sums in LDS, then adds the non-zero bins to the global 64-bit tables once. The
// counts (the weights of the first flux, equal for all) are made with the first flux in a table of 32-bit integers.
// Integer additions do not depend on their order, so the tables equal those of the Triton kernel to the bit.
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#define THREADS 256
#define MAXIMUM_BINS 4096

__device__ inline void add_local(int64_t* address, int64_t value) { atomicAdd((unsigned long long*)address, (unsigned long long)value); }

template <int PB>
__global__ void accumulate_kernel(const double* t, const double* inverse, const int64_t* quantized, const int64_t* weight,
                                  int64_t* sums, int64_t* counts, int n, int bins, int rows, int fluxes) {
    extern __shared__ char shared[];
    int64_t* local_sums = (int64_t*)shared;
    int32_t* local_counts = (int32_t*)(local_sums + bins);
    const int row = blockIdx.y;
    const int start = blockIdx.x * PB;
    const double factor = inverse[row];
    int bin_of[PB / THREADS];
    int point[PB / THREADS];
#pragma unroll
    for (int k = 0; k < PB / THREADS; k++) {
        const int i = start + k * THREADS + threadIdx.x;
        point[k] = i;
        if (i < n) {
            const double x = t[i] * factor;
            const double phase = x - floor(x);
            bin_of[k] = min(max((int)(phase * bins), 0), bins - 1);
        } else {
            bin_of[k] = -1;
        }
    }
    for (int flux = 0; flux < fluxes; flux++) {
        for (int b = threadIdx.x; b < bins; b += THREADS) {
            local_sums[b] = 0;
            if (flux == 0) local_counts[b] = 0;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < PB / THREADS; k++) {
            if (bin_of[k] >= 0) {
                add_local(&local_sums[bin_of[k]], quantized[(int64_t)flux * n + point[k]]);
                if (flux == 0) atomicAdd(&local_counts[bin_of[k]], (int32_t)weight[point[k]]);
            }
        }
        __syncthreads();
        const int64_t base = ((int64_t)flux * rows + row) * bins;
        for (int b = threadIdx.x; b < bins; b += THREADS) {
            const int64_t s = local_sums[b];
            if (s != 0) add_local(&sums[base + b], s);
            if (flux == 0) {
                const int32_t c = local_counts[b];
                if (c != 0) add_local(&counts[(int64_t)row * bins + b], (int64_t)c);
            }
        }
        __syncthreads();
    }
}

void accumulate(torch::Tensor t, torch::Tensor inverse, torch::Tensor quantized, torch::Tensor weight, torch::Tensor sums,
                torch::Tensor counts, int64_t bins, int64_t points_per_block) {
    TORCH_CHECK(t.is_cuda() && inverse.is_cuda() && quantized.is_cuda() && weight.is_cuda() && sums.is_cuda() && counts.is_cuda(), "every tensor must be on the device");
    TORCH_CHECK(t.device() == inverse.device() && t.device() == quantized.device() && t.device() == weight.device() && t.device() == sums.device() && t.device() == counts.device(),
                "every tensor must be on the device of t");
    TORCH_CHECK(t.dtype() == torch::kFloat64 && inverse.dtype() == torch::kFloat64, "t and inverse must be float64");
    TORCH_CHECK(quantized.dtype() == torch::kInt64 && weight.dtype() == torch::kInt64 && sums.dtype() == torch::kInt64 && counts.dtype() == torch::kInt64, "quantized, weight, sums and counts must be int64");
    TORCH_CHECK(t.is_contiguous() && inverse.is_contiguous() && quantized.is_contiguous() && weight.is_contiguous() && sums.is_contiguous() && counts.is_contiguous(), "every tensor must be contiguous");
    TORCH_CHECK(bins >= 1 && bins <= MAXIMUM_BINS, "bins must be from 1 to ", MAXIMUM_BINS, ", not ", bins);
    TORCH_CHECK(t.numel() >= 1 && t.numel() < (1LL << 31), "t must hold from 1 to 2^31 - 1 points");
    TORCH_CHECK(quantized.dim() == 2 && quantized.size(1) == t.numel(), "quantized must be fluxes by points, with as many points as t");
    TORCH_CHECK(weight.numel() >= t.numel(), "weight must hold a weight for every point of t");
    TORCH_CHECK(inverse.numel() >= 1 && inverse.numel() <= 2147483647LL, "inverse must hold from 1 to 2^31 - 1 periods");
    TORCH_CHECK(sums.numel() == quantized.size(0) * inverse.numel() * bins && counts.numel() == inverse.numel() * bins, "sums must be fluxes by periods by bins and counts periods by bins");
    c10::hip::HIPGuardMasqueradingAsCUDA device_guard(t.device());  // the launch is on the stream of the current device: that of the tensors
    const int n = t.numel(), rows = inverse.numel(), fluxes = quantized.size(0);
    const size_t shared = (size_t)bins * (sizeof(int64_t) + sizeof(int32_t));
    const dim3 block(THREADS), grid((n + points_per_block - 1) / points_per_block, rows);
    hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    (void)hipGetLastError();  // clears an error left by earlier work, so that the check after the launch reports this kernel's own failure and not a sticky one from before
#define LAUNCH(P) hipLaunchKernelGGL(accumulate_kernel<P>, grid, block, shared, stream, t.data_ptr<double>(), inverse.data_ptr<double>(), \
    quantized.data_ptr<int64_t>(), weight.data_ptr<int64_t>(), sums.data_ptr<int64_t>(), counts.data_ptr<int64_t>(), n, (int)bins, rows, fluxes)
    if (points_per_block == 4096) LAUNCH(4096);
    else if (points_per_block == 8192) LAUNCH(8192);
    else TORCH_CHECK(false, "points_per_block must be 4096 or 8192");
    const hipError_t launched = hipGetLastError();  // a launch that fails (a grid or a local memory over the limits of the device) leaves the sums at zero unless it is asked for
    TORCH_CHECK(launched == hipSuccess, "the accumulation kernel did not launch: ", hipGetErrorString(launched));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("accumulate", &accumulate); }
