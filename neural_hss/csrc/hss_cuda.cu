// Tensor-core kernels for the Neural-HSS backbone (any number of spatial axes).
//
// Every mode product of the CP-HSS layer, forward and backward, is a GEMM over
// a strided view of the activations, so one generic kernel covers all of them
// without permute copies:
//
//   C[z](m, n) = sum_k A[z](m, k) * B[z](k, n)      (+ bias / LeakyReLU epilogue)
//
// The batch index z and the M, N and K indices may each be "blocked"
// (index = outer * blk + inner, with separate strides for outer and inner), so
// the CP terms can be stacked along M, N, K or the batch and read or written in
// place. K blocks are padded to whole BK tiles (zero-filled), which allows any
// block length; M and N tails are masked.
//
// Math runs on TF32 tensor cores (mma.sync m16n8k8). With SPLIT3 each operand
// is split into hi + lo TF32 parts and the product is accumulated as
// a_lo*b_hi + a_hi*b_lo + a_hi*b_hi (3xTF32), which keeps FP32-level accuracy
// (~1e-6 relative error on the layer, vs ~5e-4 for plain TF32).
//
// Requires sm_80+ (cp.async, TF32 mma). Python glue: neural_hss/cuda_backbone.py.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace hss {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int WM = BM / WARPS_M;  // 64 rows per warp
constexpr int WN = BN / WARPS_N;  // 32 cols per warp
constexpr int MT = WM / 16;       // m16 tiles per warp
constexpr int NT = WN / 8;        // n8 tiles per warp
constexpr int THREADS = WARPS_M * WARPS_N * 32;
constexpr int STAGES = 3;
constexpr int VEC = BM * BK / 4 / THREADS;  // 16-byte vectors per operand per stage and thread
constexpr int64_t BIG = 1 << 30;            // block length meaning "not blocked"

// Shared-memory pitches (floats). K-contiguous tiles use BK + 4, M/N-contiguous
// tiles use 128 + 8; both make the m16n8k8 fragment loads bank-conflict free
// and keep every row 16-byte aligned for cp.async.
constexpr int PITCH_K = BK + 4;
constexpr int PITCH_MN = BM + 8;
static_assert(BM == BN, "tile pitches assume BM == BN");
static_assert(THREADS % (BK / 4) == 0 && THREADS % (BM / 4) == 0, "vector layout assumes whole rows per pass");

enum Epilogue : int { EPI_STORE = 0, EPI_BIAS_ACT = 1 };

// Division by a divisor fixed at launch, as a multiply-high and a shift
// (Granlund & Montgomery; same scheme as PyTorch's IntDivider). Valid for n < 2^31.
struct FastDiv {
  uint32_t d, mul, shift;
};

inline FastDiv make_fastdiv(int64_t d) {
  TORCH_CHECK(d >= 1 && d <= BIG, "divisor out of range: ", d);
  FastDiv f;
  f.d = static_cast<uint32_t>(d);
  uint32_t s = 0;
  while ((uint64_t{1} << s) < f.d) ++s;
  f.shift = s;
  const uint64_t magic = ((uint64_t{1} << 32) * ((uint64_t{1} << s) - f.d)) / f.d + 1;
  TORCH_CHECK(magic <= 0xffffffffull, "fastdiv magic overflow");
  f.mul = static_cast<uint32_t>(magic);
  return f;
}

__device__ __forceinline__ int fdiv(int n, const FastDiv& f) {
  const uint32_t u = static_cast<uint32_t>(n);
  return static_cast<int>((__umulhi(u, f.mul) + u) >> f.shift);
}

// (i % blk) * s + (i / blk) * sb
__device__ __forceinline__ long long blocked(int i, const FastDiv& blk, long long s, long long sb) {
  const int q = fdiv(i, blk);
  return static_cast<long long>(i - q * static_cast<int>(blk.d)) * s + static_cast<long long>(q) * sb;
}

// (i / div) % mod
__device__ __forceinline__ int div_mod(int i, const FastDiv& div, const FastDiv& mod) {
  const int q = fdiv(i, div);
  return q - fdiv(q, mod) * static_cast<int>(mod.d);
}

// One GEMM operand. Element (z, i, k) of A (i = m) or (z, k, i) of B (i = n) is at
//   ptr + (z % zblk)*sz + (z / zblk)*szb + (i % blk)*s + (i / blk)*sb + (k % kblk)*sk + (k / kblk)*skb
struct Operand {
  const float* ptr;
  long long sz, szb, s, sb, sk, skb;
  FastDiv blk;
};

struct Params {
  int M, N, K;
  int kblk, tpb, ktiles;           // K block length (shared by A and B), BK tiles per block, total K tiles
  int batch, splits, tps;          // batch count, K splits, K tiles per split
  int n_tiles, tiles, items;       // output tiles along N, per (z, split), and work items in total
  FastDiv f_tpb, f_ntiles, f_tiles, f_splits, f_zblk;
  Operand a, b;
  // C(z, split, m, n) = c + (z % zblk)*c_sz + (z / zblk)*c_szb + split*c_ssplit
  //                       + (m % mblk)*c_sm + (m / mblk)*c_smb + (n % nblk) + (n / nblk)*c_snb
  float* c;            // may be null for EPI_BIAS_ACT (inference: pre-activation not stored)
  float* c2;           // EPI_BIAS_ACT: activated output, same addressing as c (null: no activation)
  const float* bias;   // EPI_BIAS_ACT: bias[m], or bias[(n / bias_div) % bias_mod] when bias_on_n (null: none)
  const float* slope;  // EPI_BIAS_ACT with c2: LeakyReLU slope (device scalar)
  long long c_sz, c_szb, c_ssplit, c_sm, c_smb, c_snb;
  FastDiv c_mblk, c_nblk;
  int bias_on_n;
  FastDiv bias_div, bias_mod;
};

__device__ __forceinline__ void cp_async16(float* smem, const float* gmem, bool pred) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const int bytes = pred ? 16 : 0;  // 0 source bytes -> zero fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Round to the nearest TF32 value (ties away from zero), like cvt.rna.tf32.f32
// for finite inputs. sm_80 has no native instruction for that cvt and ptxas
// emulates it with ~4 instructions; this is 2 (IADD3 + LOP3).
__device__ __forceinline__ uint32_t to_tf32(float x) {
  return (__float_as_uint(x) + 0x1000u) & 0xffffe000u;
}

// Small part of the 3xTF32 split, x - hi. As in CUTLASS's FastF32 it is not
// rounded: the tensor core ignores its low 13 mantissa bits (round toward zero).
// Measured: same accuracy as rounding it, two instructions fewer per value.
__device__ __forceinline__ uint32_t lo_part(float x, uint32_t hi) {
  return __float_as_uint(x - __uint_as_float(hi));
}

// Not volatile: the compiler may interleave independent MMAs to hide their latency.
__device__ __forceinline__ void mma_tf32(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// d = a b (zero accumulator)
__device__ __forceinline__ void mma_tf32_zero(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.f));
}


// A_KC: A is K-contiguous (a.sk == 1), else M-contiguous (a.s == 1).
// B_KC: B is K-contiguous (b.sk == 1), else N-contiguous (b.s == 1).
//
// Persistent: each CTA walks work items (output tile, batch, K split) with a
// stride of gridDim.x. The cp.async pipeline runs over the flattened sequence of
// K tiles of all its items, so the next item's first tiles are already loading
// while the current item finishes and stores its results.
template <bool A_KC, bool B_KC, int EPI, bool SPLIT3>
__global__ void __launch_bounds__(THREADS) gemm_kernel(const __grid_constant__ Params p) {
  constexpr int A_STAGE = A_KC ? BM * PITCH_K : BK * PITCH_MN;
  constexpr int B_STAGE = B_KC ? BN * PITCH_K : BK * PITCH_MN;
  extern __shared__ __align__(16) float smem[];
  float* As = smem;
  float* Bs = smem + STAGES * A_STAGE;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int g = lane >> 2, t4 = lane & 3;
  if (static_cast<int>(blockIdx.x) >= p.items) return;

  struct Item {
    int m0, n0, z, split, t0, ktiles;
  };
  auto item = [&](int w) {
    Item it;
    const int rest = fdiv(w, p.f_tiles);
    const int tile = w - rest * p.tiles;
    const int mt = fdiv(tile, p.f_ntiles);
    it.m0 = mt * BM;
    it.n0 = (tile - mt * p.n_tiles) * BN;
    it.z = fdiv(rest, p.f_splits);
    it.split = rest - it.z * p.splits;
    it.t0 = it.split * p.tps;
    it.ktiles = min(p.ktiles - it.t0, p.tps);
    return it;
  };
  auto zoff = [&](int z, long long sz, long long szb) {
    const int q = fdiv(z, p.f_zblk);
    return static_cast<long long>(z - q * static_cast<int>(p.f_zblk.d)) * sz + static_cast<long long>(q) * szb;
  };

  // ---- Load side. Vector v = tid + i*THREADS of a stage sits at (row, col) of the
  // tile: K-contiguous tiles have rows = m/n and BK/4 vectors per row, the
  // others rows = k and BM/4 vectors per row. Its K position inside the tile is
  // the column (same for every i) or the row.
  const int kc_pos = (tid % (BK / 4)) * 4;
  const int kr_pos = tid / (BM / 4);
  constexpr int KR_STEP = THREADS / (BM / 4);
  const float* a_src[VEC];
  const float* b_src[VEC];
  bool a_ok[VEC], b_ok[VEC];
  int kb = 0, ki = 0;  // K cursor: block index and offset inside the block
  int wl = blockIdx.x, kl = 0, kl_end = 0;
  auto setup_load = [&](int w) {
    const Item it = item(w);
    kl = 0;
    kl_end = it.ktiles;
    kb = fdiv(it.t0, p.f_tpb);
    ki = (it.t0 - kb * p.tpb) * BK;
    const float* a0 = p.a.ptr + zoff(it.z, p.a.sz, p.a.szb);
    const float* b0 = p.b.ptr + zoff(it.z, p.b.sz, p.b.szb);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const int v = tid + i * THREADS;
      if constexpr (A_KC) {
        const int m = it.m0 + v / (BK / 4);
        a_ok[i] = m < p.M;
        a_src[i] = a0 + (a_ok[i] ? blocked(m, p.a.blk, p.a.s, p.a.sb) : 0) + kc_pos;
      } else {
        const int m = it.m0 + (v % (BM / 4)) * 4;
        a_ok[i] = m < p.M;
        a_src[i] = a0 + (a_ok[i] ? blocked(m, p.a.blk, 1, p.a.sb) : 0) + static_cast<long long>(v / (BM / 4)) * p.a.sk;
      }
      if constexpr (B_KC) {
        const int n = it.n0 + v / (BK / 4);
        b_ok[i] = n < p.N;
        b_src[i] = b0 + (b_ok[i] ? blocked(n, p.b.blk, p.b.s, p.b.sb) : 0) + kc_pos;
      } else {
        const int n = it.n0 + (v % (BN / 4)) * 4;
        b_ok[i] = n < p.N;
        b_src[i] = b0 + (b_ok[i] ? blocked(n, p.b.blk, 1, p.b.sb) : 0) + static_cast<long long>(v / (BN / 4)) * p.b.sk;
      }
    }
  };
  // Shared-memory destination of vector v.
  auto dst = [](int v, bool kc) {
    return kc ? (v / (BK / 4)) * PITCH_K + (v % (BK / 4)) * 4 : (v / (BM / 4)) * PITCH_MN + (v % (BM / 4)) * 4;
  };
  // Issue the next K tile of the flattened sequence (if any); always commits a group.
  auto load_next = [&](int stage) {
    if (wl < p.items) {
      const long long a_k = static_cast<long long>(kb) * p.a.skb + static_cast<long long>(ki) * p.a.sk;
      const long long b_k = static_cast<long long>(kb) * p.b.skb + static_cast<long long>(ki) * p.b.sk;
      float* as = As + stage * A_STAGE;
      float* bs = Bs + stage * B_STAGE;
      const bool kc_ok = ki + kc_pos < p.kblk;  // K-contiguous vectors: kblk % 4 == 0, all-or-nothing
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const int v = tid + i * THREADS;
        const bool kr_ok = ki + kr_pos + i * KR_STEP < p.kblk;
        const bool oa = a_ok[i] && (A_KC ? kc_ok : kr_ok);
        const bool ob = b_ok[i] && (B_KC ? kc_ok : kr_ok);
        cp_async16(as + dst(v, A_KC), oa ? a_src[i] + a_k : p.a.ptr, oa);
        cp_async16(bs + dst(v, B_KC), ob ? b_src[i] + b_k : p.b.ptr, ob);
      }
      ki += BK;
      if (ki >= p.kblk) ki = 0, ++kb;
      if (++kl == kl_end) {
        wl += gridDim.x;
        if (wl < p.items) setup_load(wl);
      }
    }
    cp_async_commit();
  };

  // ---- Compute side.
  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

  float alpha = 0.f;
  if constexpr (EPI == EPI_BIAS_ACT) {
    if (p.c2 != nullptr) alpha = *p.slope;
  }

  auto epilogue = [&](const Item& it) {
    // Each thread owns rows (g, g+8) of every m16 tile and column pairs of every n8 tile.
    const long long c_z = zoff(it.z, p.c_sz, p.c_szb) + static_cast<long long>(it.split) * p.c_ssplit;
    long long col_off[NT];
    bool col_ok[NT];
    float2 col_bias[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int n = it.n0 + wn * WN + j * 8 + t4 * 2;
      col_ok[j] = n < p.N;  // N and c_nblk are even, so n + 1 is valid too
      col_off[j] = col_ok[j] ? blocked(n, p.c_nblk, 1, p.c_snb) : 0;
      col_bias[j] = make_float2(0.f, 0.f);
      if constexpr (EPI == EPI_BIAS_ACT) {
        if (p.bias != nullptr && p.bias_on_n && col_ok[j])
          col_bias[j] = make_float2(p.bias[div_mod(n, p.bias_div, p.bias_mod)],
                                    p.bias[div_mod(n + 1, p.bias_div, p.bias_mod)]);
      }
    }
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = it.m0 + wm * WM + i * 16 + g + h * 8;
        if (m >= p.M) continue;
        const long long row = c_z + blocked(m, p.c_mblk, p.c_sm, p.c_smb);
        float row_bias = 0.f;
        if constexpr (EPI == EPI_BIAS_ACT) {
          if (p.bias != nullptr && !p.bias_on_n) row_bias = p.bias[m];
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          if (!col_ok[j]) continue;
          float2 val = make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
          if constexpr (EPI == EPI_BIAS_ACT) {
            val.x += row_bias + col_bias[j].x;
            val.y += row_bias + col_bias[j].y;
            if (p.c != nullptr) *reinterpret_cast<float2*>(p.c + row + col_off[j]) = val;
            if (p.c2 != nullptr) {
              const float2 act =
                  make_float2(val.x >= 0.f ? val.x : alpha * val.x, val.y >= 0.f ? val.y : alpha * val.y);
              *reinterpret_cast<float2*>(p.c2 + row + col_off[j]) = act;
            }
          } else {
            *reinterpret_cast<float2*>(p.c + row + col_off[j]) = val;
          }
        }
      }
    }
  };

  setup_load(wl);
  int wc = blockIdx.x, kc = 0;
  Item ic = item(wc);

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) load_next(s);

  // Tensor cores round the accumulator toward zero, so a long sum kept inside the
  // MMA drifts by up to one ulp per step: ~K/8 ulps in total (1.7e-4 relative
  // error at K = 16384, worse than plain TF32). With SPLIT3 every BK tile is
  // therefore summed from zero in `tacc` and added to `acc` with round-to-nearest
  // FP32 adds, which keeps the error at the FP32 level (~3e-7) for any K.
  constexpr bool FLUSH = SPLIT3;
  for (int flat = 0; wc < p.items; ++flat) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    load_next((flat + STAGES - 1) % STAGES);
    const float* as = As + (flat % STAGES) * A_STAGE;
    const float* bs = Bs + (flat % STAGES) * B_STAGE;
    float tacc[MT][NT][4];  // FLUSH: this BK tile's partial sums
#pragma unroll
    for (int kk = 0; kk < BK; kk += 8) {
      uint32_t bh[NT][2], bl[NT][2];
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int n = wn * WN + j * 8 + g;
        float v0, v1;
        if constexpr (B_KC) {
          v0 = bs[n * PITCH_K + kk + t4];
          v1 = bs[n * PITCH_K + kk + t4 + 4];
        } else {
          v0 = bs[(kk + t4) * PITCH_MN + n];
          v1 = bs[(kk + t4 + 4) * PITCH_MN + n];
        }
        bh[j][0] = to_tf32(v0);
        bh[j][1] = to_tf32(v1);
        if constexpr (SPLIT3) {
          bl[j][0] = lo_part(v0, bh[j][0]);
          bl[j][1] = lo_part(v1, bh[j][1]);
        }
      }
      uint32_t ah[MT][4], al[MT][4];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int m = wm * WM + i * 16 + g;
        float v[4];
        if constexpr (A_KC) {
          v[0] = as[m * PITCH_K + kk + t4];
          v[1] = as[(m + 8) * PITCH_K + kk + t4];
          v[2] = as[m * PITCH_K + kk + t4 + 4];
          v[3] = as[(m + 8) * PITCH_K + kk + t4 + 4];
        } else {
          v[0] = as[(kk + t4) * PITCH_MN + m];
          v[1] = as[(kk + t4) * PITCH_MN + m + 8];
          v[2] = as[(kk + t4 + 4) * PITCH_MN + m];
          v[3] = as[(kk + t4 + 4) * PITCH_MN + m + 8];
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          ah[i][q] = to_tf32(v[q]);
          if constexpr (SPLIT3) al[i][q] = lo_part(v[q], ah[i][q]);
        }
      }
      // Each pass touches all MT*NT accumulators once, so consecutive MMAs are
      // independent; small correction terms go first.
      auto& dst_acc = FLUSH ? tacc : acc;
      if constexpr (SPLIT3) {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            if (kk == 0) {
              mma_tf32_zero(dst_acc[i][j], al[i], bh[j]);
            } else {
              mma_tf32(dst_acc[i][j], al[i], bh[j]);
            }
          }
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
          for (int j = 0; j < NT; ++j) mma_tf32(dst_acc[i][j], ah[i], bl[j]);
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) mma_tf32(dst_acc[i][j], ah[i], bh[j]);
    }
    if constexpr (FLUSH) {
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] += tacc[i][j][q];
    }
    if (++kc == ic.ktiles) {
      epilogue(ic);
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      wc += gridDim.x;
      kc = 0;
      if (wc < p.items) ic = item(wc);
    }
  }
  cp_async_wait<0>();
}

// dY = dZ * act'(Y) for the trainable LeakyReLU, plus per-row sums for the bias
// gradient and the slope gradient, over contiguous rows of length L. A block of
// AB_WARPS warps covers AB_WARPS / WPR rows with WPR warps per row: WPR = 1 for
// short rows (the last spatial axis of a 3D field), WPR = AB_WARPS for long ones.
constexpr int AB_WARPS = 8;

template <int WPR>
__global__ void __launch_bounds__(AB_WARPS * 32)
    act_backward_kernel(const float* __restrict__ dz, const float* __restrict__ y, const float* __restrict__ slope,
                        float* __restrict__ dy, float* __restrict__ row_dbias, float* __restrict__ row_dslope, int L,
                        long long rows) {
  static_assert(WPR == 1 || WPR == AB_WARPS, "one warp or one block per row");
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long long row = static_cast<long long>(blockIdx.x) * (AB_WARPS / WPR) + warp / WPR;
  const float alpha = *slope;
  float s = 0.f, sa = 0.f;
  if (row < rows) {
    const long long base = row * L;
    const float4* dz4 = reinterpret_cast<const float4*>(dz + base);
    const float4* y4 = reinterpret_cast<const float4*>(y + base);
    float4* dy4 = reinterpret_cast<float4*>(dy + base);
    for (int i = (warp % WPR) * 32 + lane; i < L / 4; i += WPR * 32) {
      const float4 a = dz4[i], v = y4[i];
      float4 d;
      d.x = v.x >= 0.f ? a.x : alpha * a.x;
      d.y = v.y >= 0.f ? a.y : alpha * a.y;
      d.z = v.z >= 0.f ? a.z : alpha * a.z;
      d.w = v.w >= 0.f ? a.w : alpha * a.w;
      sa += (v.x < 0.f ? a.x * v.x : 0.f) + (v.y < 0.f ? a.y * v.y : 0.f) + (v.z < 0.f ? a.z * v.z : 0.f) +
            (v.w < 0.f ? a.w * v.w : 0.f);
      s += (d.x + d.y) + (d.z + d.w);
      dy4[i] = d;
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    s += __shfl_xor_sync(0xffffffff, s, o);
    sa += __shfl_xor_sync(0xffffffff, sa, o);
  }
  if constexpr (WPR == 1) {
    if (lane == 0 && row < rows) {
      row_dbias[row] = s;
      row_dslope[row] = sa;
    }
  } else {
    __shared__ float red[2][AB_WARPS];
    if (lane == 0) {
      red[0][warp] = s;
      red[1][warp] = sa;
    }
    __syncthreads();
    if (warp == 0) {
      s = lane < AB_WARPS ? red[0][lane] : 0.f;
      sa = lane < AB_WARPS ? red[1][lane] : 0.f;
      for (int o = AB_WARPS / 2; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, o);
        sa += __shfl_xor_sync(0xffffffff, sa, o);
      }
      if (lane == 0 && row < rows) {
        row_dbias[row] = s;
        row_dslope[row] = sa;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Structured mode product: an HSS factor applied block by block, without the
// dense matrix.
//
//   C[z](m, o) = epilogue( sum_t sum_i X[z](m, t I + i) W_t(o, i) )
//
// m runs over fibers, o over all O <= 128 outputs of a fiber and i over its I
// inputs. With sum_terms the CP terms t are summed into one output; otherwise z
// is the term. W_t is a HODLR matrix with uniform splits: NL = 2^L leaves of
// ro = O / NL rows and ci = I / NL columns, and for every node at level l < L the
// off-diagonal blocks U12 V12^T (rows: first half, columns: second half) and
// U21 V21^T, of rank r. Per fiber:
//
//   1. each input segment j (the columns of leaf j) goes through the leaf into
//      the outputs of segment j, and is projected on the V rows of the L blocks
//      whose columns contain it: L r partial projections per segment;
//   2. for each output segment s and level l, the block's projection is the sum
//      of the partials of the sibling half of s's node, and is multiplied by the
//      U rows of s.
//
// That is O ci + (I + O) L r multiply-adds per fiber instead of O I (6144 vs
// 16384 at O = I = 128, L = 2, r = 4); the result is the same matrix product up
// to rounding.
//
// A CTA computes HB_F = 64 fibers x all outputs, one CP term at a time (a unit),
// with the whole input of the term in shared memory (double buffered: the next
// unit loads while this one computes). Its 8 warps are NL columns x 8 / NL rows:
// warp column s owns output segment s, so it needs only input segment s for the
// leaf and its own partials, and every warp works on its own leaf at the same
// time, MT x NT independent accumulators as in gemm_kernel. The partials are then
// exchanged through shared memory for the expansions. The host (cuda_backbone.py)
// packs the factors of every term, and the CTA keeps all of them in shared memory:
//   D: O rows (outputs) x ci (the leaf of the row's segment), pitch dp = ci + 4
//   P: NL R rows (segment j, column l r + t) x ci (the V rows of segment j), pitch dp
//   E: O rows x R (the U rows of the row's segment, column l r + t), pitch ep = R + 4
// with R = L r rounded up to a multiple of 8 (zero padding). The input gradient
// uses the same kernel on the packs of the transposed factors.
constexpr int HB_WARPS = THREADS / 32;
constexpr int HB_F = 64;     // fibers per tile
constexpr int HB_OMAX = 128; // outputs per fiber
constexpr int HB_PMAX = 64;  // partial projections per fiber: NL R

struct HssParams {
  int M, O, I, terms, sum_terms;
  int NL, L, r, R, ro, ci;
  int dp, ep, pp, xp;  // pitches: D and P rows, E rows, partial rows, X tile rows
  int xtile;           // floats per X tile
  int fstride;         // floats per term in the packed factors
  int m_tiles, items;
  FastDiv f_mtiles, f_r, f_i4;
  // X(z, m, k), k = t I + i: x + z x_sz + (m % blk) x_s + (m / blk) x_sb + i x_sk + t x_skb
  const float* x;
  long long x_sz, x_s, x_sb, x_sk, x_skb;
  FastDiv x_blk;
  const float* fact;
  // C(z, m, o): c + z c_sz + (m % mblk) c_sm + (m / mblk) c_smb + o c_sn
  float* c;            // EPI_BIAS_ACT: the pre-activation (may be null)
  float* c2;           // EPI_BIAS_ACT: LeakyReLU of it (null: no activation)
  const float* bias;   // EPI_BIAS_ACT: bias[(o or m / bias_div) % bias_mod] (null: none)
  const float* slope;  // EPI_BIAS_ACT with c2
  long long c_sz, c_sm, c_smb, c_sn;
  FastDiv c_mblk;
  int bias_on_n;
  FastDiv bias_div, bias_mod;
  // The routed projections (the A operand of the expansions), saved for the
  // factor gradient (PJ > 0): proj + term pj_sz + m NL R + j R + 8 q + s(c) for
  // segment j, column 8 q + c of its R, with s(c) = 2 (c % 4) + c / 4 (the order
  // of the kernel's float2 stores).
  float* proj;
  long long pj_sz;
};

// LAST: X and C are contiguous along the factor's axis (the last axis of the
// field: x_sk = 1, c_sn = 1); else along the fibers (x_s = 1, c_sm = 1).
// PJ: 0 = the product, 1 = the product and its routed projections (proj),
// 2 = the routed projections only (no leaves, expansions or output).
template <bool LAST, int EPI, bool SPLIT3, int NL, int PJ>
__global__ void __launch_bounds__(THREADS) hss_kernel(const __grid_constant__ HssParams p) {
  constexpr int WROWS = HB_WARPS / NL;      // warp rows
  constexpr int MT = HB_F / 16 / WROWS;     // m16 tiles per warp
  constexpr int NTM = HB_OMAX / NL / 8;     // n8 output tiles per warp, at most
  constexpr int PTM = HB_PMAX / NL / 8;     // n8 partial tiles per warp, at most
  static_assert(WROWS * NL == HB_WARPS && MT * 16 * WROWS == HB_F, "warp layout");
  extern __shared__ __align__(16) float smem[];
  float* Xs = smem;                               // two X tiles
  float* Fs = smem + 2 * p.xtile;                 // the factors of all terms
  float* Ps = Fs + p.terms * p.fstride;           // partials: [fiber][segment R + column]
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int wn = warp % NL;                       // warp column: the segment
  const int f0 = (warp / NL) * MT * 16;           // the warp's first fiber in the tile
  const int nt = p.ro / 8, pt = p.R / 8;
  if (static_cast<int>(blockIdx.x) >= p.items) return;

  for (int v = tid; v < p.terms * p.fstride / 4; v += THREADS)
    reinterpret_cast<float4*>(Fs)[v] = __ldg(reinterpret_cast<const float4*>(p.fact) + v);

  auto item_z = [&](int w) { return p.sum_terms ? 0 : fdiv(w, p.f_mtiles); };
  // The whole input of one unit (item w, term t) into tile `buf`; commits a group.
  auto load_unit = [&](int w, int t, int buf) {
    const int z = item_z(w);
    const int m0 = (w - z * p.m_tiles) * HB_F;
    const float* x0 = p.x + static_cast<long long>(z) * p.x_sz + static_cast<long long>(t) * p.x_skb;
    float* xs = Xs + buf * p.xtile;
    const int nvec = HB_F * p.I / 4;
    for (int v = tid; v < nvec; v += THREADS) {
      if constexpr (LAST) {  // tile [fiber][k], vectors along k
        const int f = fdiv(v, p.f_i4);
        const int k = (v - f * (p.I / 4)) * 4;
        const int m = m0 + f;
        const bool ok = m < p.M;
        cp_async16(xs + f * p.xp + k, ok ? x0 + blocked(m, p.x_blk, p.x_s, p.x_sb) + k : p.x, ok);
      } else {  // tile [k][fiber], vectors along the fibers
        const int k = v / (HB_F / 4);
        const int f = (v % (HB_F / 4)) * 4;
        const int m = m0 + f;
        const bool ok = m < p.M;
        cp_async16(xs + k * p.xp + f,
                   ok ? x0 + blocked(m, p.x_blk, 1, p.x_sb) + static_cast<long long>(k) * p.x_sk : p.x, ok);
      }
    }
    cp_async_commit();
  };

  // As gemm_kernel's FLUSH: with SPLIT3 each term is summed from zero in `tacc`
  // and added to `acc` in FP32 (the tensor cores round toward zero).
  float acc[MT][NTM][4], tacc[MT][NTM][4], pacc[MT][PTM][4];
#pragma unroll
  for (int i = 0; i < MT; ++i) {
#pragma unroll
    for (int j = 0; j < NTM; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = tacc[i][j][q] = 0.f;
#pragma unroll
    for (int j = 0; j < PTM; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) pacc[i][j][q] = 0.f;
  }
  auto& dacc = SPLIT3 ? tacc : acc;
  float alpha = 0.f;
  if constexpr (EPI == EPI_BIAS_ACT) {
    if (p.c2 != nullptr) alpha = *p.slope;
  }

  auto split = [](float v, uint32_t& h, uint32_t& l) {
    h = to_tf32(v);
    if constexpr (SPLIT3) l = lo_part(v, h);
  };
  // B fragment from a factor stored [n][k]: row = &F[(n0 + g) * pitch + k0]
  auto frag_b = [&](const float* row, uint32_t (&bh)[2], uint32_t (&bl)[2]) {
    split(row[t4], bh[0], bl[0]);
    split(row[t4 + 4], bh[1], bl[1]);
  };
  auto mma3 = [&](float (&d)[4], const uint32_t (&ah)[4], const uint32_t (&al)[4], const uint32_t (&bh)[2],
                  const uint32_t (&bl)[2]) {
    if constexpr (SPLIT3) {
      mma_tf32(d, al, bh);
      mma_tf32(d, ah, bl);
    }
    mma_tf32(d, ah, bh);
  };

  constexpr bool out = PJ != 2;
  auto compute_unit = [&](int buf, int term, int m0) {
    const float* xs = Xs + buf * p.xtile;
    const float* Fd = Fs + term * p.fstride + (wn * p.ro + g) * p.dp;  // this warp's leaf rows
    const float* Fp = Fs + term * p.fstride + (p.O + wn * p.R + g) * p.dp;
    const float* Fe = Fs + term * p.fstride + (p.O + p.NL * p.R) * p.dp + (wn * p.ro + g) * p.ep;
    // 1. leaf and partial projections of input segment wn
    for (int k8 = 0; k8 < p.ci; k8 += 8) {
      const int k = wn * p.ci + k8;
      uint32_t ah[MT][4], al[MT][4];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int f = f0 + i * 16 + g;
        float v[4];
        if constexpr (LAST) {
          v[0] = xs[f * p.xp + k + t4];
          v[1] = xs[(f + 8) * p.xp + k + t4];
          v[2] = xs[f * p.xp + k + t4 + 4];
          v[3] = xs[(f + 8) * p.xp + k + t4 + 4];
        } else {
          v[0] = xs[(k + t4) * p.xp + f];
          v[1] = xs[(k + t4) * p.xp + f + 8];
          v[2] = xs[(k + t4 + 4) * p.xp + f];
          v[3] = xs[(k + t4 + 4) * p.xp + f + 8];
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) split(v[q], ah[i][q], al[i][q]);
      }
      if constexpr (out) {
#pragma unroll
        for (int j = 0; j < NTM; ++j) {
          if (j < nt) {
            uint32_t bh[2], bl[2];
            frag_b(Fd + j * 8 * p.dp + k8, bh, bl);
#pragma unroll
            for (int i = 0; i < MT; ++i) mma3(dacc[i][j], ah[i], al[i], bh, bl);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < PTM; ++j) {
        if (j < pt) {
          uint32_t bh[2], bl[2];
          frag_b(Fp + j * 8 * p.dp + k8, bh, bl);
#pragma unroll
          for (int i = 0; i < MT; ++i) mma3(pacc[i][j], ah[i], al[i], bh, bl);
        }
      }
    }
    // 2. partials to shared memory, for every warp column
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < PTM; ++j) {
        if (j < pt) {
          float* r0 = Ps + (f0 + i * 16 + g) * p.pp + wn * p.R + j * 8 + 2 * t4;
          *reinterpret_cast<float2*>(r0) = make_float2(pacc[i][j][0], pacc[i][j][1]);
          *reinterpret_cast<float2*>(r0 + 8 * p.pp) = make_float2(pacc[i][j][2], pacc[i][j][3]);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) pacc[i][j][q] = 0.f;
      }
    __syncthreads();
    // 3. expansions of output segment wn: block projections (sums of the sibling
    //    half's partials, per level) times the U rows
    const int lr = p.L * p.r;
    for (int qc = 0; qc < pt; ++qc) {
      uint32_t ah[MT][4], al[MT][4];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        float sums[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {  // the mma A order: (g, c), (g + 8, c), (g, c + 4), (g + 8, c + 4)
          const int col = qc * 8 + t4 + (e >> 1) * 4;
          float sum = 0.f;
          if (col < lr) {
            const int sh = p.L - 1 - fdiv(col, p.f_r);  // the column's level, counted from the leaves
            const int lo = ((wn >> sh) ^ 1) << sh;     // first segment of the sibling half
            const float* src = Ps + (f0 + i * 16 + g + (e & 1) * 8) * p.pp + col;
            for (int j = lo; j < lo + (1 << sh); ++j) sum += src[j * p.R];
          }
          if constexpr (PJ != 0) sums[e] = sum;
          split(sum, ah[i][e], al[i][e]);
        }
        if constexpr (PJ != 0) {  // columns c and c + 4 of fibers g and g + 8, as float2 (see HssParams::proj)
          const int nlr = p.NL * p.R;
          float* pj = p.proj + term * p.pj_sz + static_cast<long long>(m0 + f0) * nlr + wn * p.R + 2 * t4;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int f = i * 16 + g + h * 8;
            if (m0 + f0 + f < p.M)
              *reinterpret_cast<float2*>(pj + static_cast<long long>(f) * nlr + qc * 8) =
                  make_float2(sums[h], sums[h + 2]);
          }
        }
      }
      if constexpr (out) {
#pragma unroll
        for (int j = 0; j < NTM; ++j) {
          if (j < nt) {
            uint32_t bh[2], bl[2];
            frag_b(Fe + j * 8 * p.ep + qc * 8, bh, bl);
#pragma unroll
            for (int i = 0; i < MT; ++i) mma3(dacc[i][j], ah[i], al[i], bh, bl);
          }
        }
      }
    }
    if constexpr (SPLIT3) {
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NTM; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            acc[i][j][q] += tacc[i][j][q];
            tacc[i][j][q] = 0.f;
          }
    }
  };

  auto epilogue = [&](int w) {
    const int z = item_z(w);
    const int m0 = (w - z * p.m_tiles) * HB_F;
    const long long cz = static_cast<long long>(z) * p.c_sz;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = m0 + f0 + i * 16 + g + h * 8;
        if (m >= p.M) continue;
        const long long row = cz + blocked(m, p.c_mblk, p.c_sm, p.c_smb);
        float mbias = 0.f;
        if constexpr (EPI == EPI_BIAS_ACT) {
          if (p.bias != nullptr && !p.bias_on_n) mbias = p.bias[div_mod(m, p.bias_div, p.bias_mod)];
        }
#pragma unroll
        for (int j = 0; j < NTM; ++j) {
          if (j >= nt) continue;
          const int n = wn * p.ro + j * 8 + 2 * t4;
          float2 val = make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
          auto store = [&](float* base, float2 v) {
            if constexpr (LAST) {
              *reinterpret_cast<float2*>(base + row + n) = v;
            } else {
              base[row + static_cast<long long>(n) * p.c_sn] = v.x;
              base[row + static_cast<long long>(n + 1) * p.c_sn] = v.y;
            }
          };
          if constexpr (EPI == EPI_BIAS_ACT) {
            if (p.bias != nullptr) {
              if (p.bias_on_n) {
                val.x += p.bias[div_mod(n, p.bias_div, p.bias_mod)];
                val.y += p.bias[div_mod(n + 1, p.bias_div, p.bias_mod)];
              } else {
                val.x += mbias;
                val.y += mbias;
              }
            }
            if (p.c != nullptr) store(p.c, val);
            if (p.c2 != nullptr)
              store(p.c2, make_float2(val.x >= 0.f ? val.x : alpha * val.x, val.y >= 0.f ? val.y : alpha * val.y));
          } else {
            store(p.c, val);
          }
        }
      }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NTM; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
  };

  // Units: (item, term) when the terms are summed, else items; this CTA takes
  // items blockIdx.x + k gridDim.x. The next unit loads while this one computes.
  const int upi = p.sum_terms ? p.terms : 1;
  int w = blockIdx.x, t = 0;
  load_unit(w, 0, 0);
  for (int u = 0; w < p.items; ++u) {
    cp_async_wait<0>();
    __syncthreads();  // this unit's tile is in; every warp is done with the previous unit
    int w_next = w, t_next = t + 1;
    if (t_next == upi) {
      t_next = 0;
      w_next = w + gridDim.x;
    }
    if (w_next < p.items) load_unit(w_next, t_next, (u + 1) & 1);
    if constexpr (PJ == 0) {
      compute_unit(u & 1, p.sum_terms ? t : item_z(w), 0);
    } else {
      const int z = item_z(w);
      compute_unit(u & 1, p.sum_terms ? t : z, (w - z * p.m_tiles) * HB_F);
    }
    if (out && t == upi - 1) epilogue(w);
    w = w_next;
    t = t_next;
  }
  cp_async_wait<0>();
}

// ---------------------------------------------------------------------------
// Structured factor gradient: the gradient of the HSS parameters of a mode
// product C(m, o) = sum_i X(m, i) W_t(o, i), from X, G = dL/dC and the routed
// projections that hss_kernel saved (HssParams::proj): P of the product (V^T X,
// routed to the output segments) and Q of the input-gradient product (U^T G,
// routed to the input segments). For segment j:
//
//   dD_j = sum_m G_j(m)^T X_j(m)   (ro x ci: leaf j)
//   dU_j = sum_m G_j(m)^T P_j(m)   (ro x R: the U rows of output segment j, column l r + t)
//   dV_j = sum_m X_j(m)^T Q_j(m)   (ci x R: the V rows of input segment j)
//
// that is block-diagonal GEMMs with the fibers as the reduction, O ci + (O + I) R
// multiply-adds per fiber instead of O I, and no routing. CTA (term t, k) reduces
// the fiber tiles k, k + cpt, ... of term t (HW_F fibers each, HW_STAGES tiles in
// flight) and writes its sums, in the packed layout of the forward factors (D,
// then the V^T rows of P, then the U rows of E), to ws[t][k]; the host sums the
// CTAs. Warp (segment j, fiber split ks, row split rs) owns the 16-row blocks rs,
// rs + RS, ... of dD_j, dU_j and dV_j, over the k-steps ks, ks + KS, ... of every
// tile. With SPLIT3, every tile is summed from zero and added in FP32 (the tensor
// cores round toward zero, and the reduction is long).
constexpr int HW_F = 32;      // fibers per tile
constexpr int HW_STAGES = 3;  // tiles in flight

struct HssWParams {
  int M, O, I, terms;
  int NL, R, ro, ci, nlr;   // nlr = NL R: projections per fiber
  int dp, ep, fstride;      // packed layout (as HssParams)
  int xp, gp, pq;           // pitches of the X, G and projection tiles
  int xtile, gtile, ptile;  // floats per tile
  int m_tiles, cpt;         // fiber tiles per term, CTAs per term
  FastDiv f_i4, f_o4, f_p4;
  const float* x;           // X(t, m, i): x + t x_sz + (m % blk) x_s + (m / blk) x_sb + i x_sk
  long long x_sz, x_s, x_sb, x_sk;
  FastDiv x_blk;
  const float* gr;          // G(t, m, o), same addressing
  long long g_sz, g_s, g_sb, g_sk;
  FastDiv g_blk;
  const float* pr;          // P(t, m, .) = pr + t pj_sz + m nlr, laid out as HssParams::proj
  const float* qr;          // Q(t, m, .), the same
  long long pj_sz;
  float* ws;
};

// Warp layout of hss_wgrad_kernel. NL = 4: the two warps of a segment split the
// k-steps of a tile and each owns all 16-row blocks of its leaf (twice the
// independent accumulators of a row split); their sums are added at the end.
template <int NL>
struct WgradLayout {
  static constexpr int WPS = HB_WARPS / NL;     // warps per segment
  static constexpr int NDT = HB_OMAX / NL / 8;  // n8 tiles of a leaf's columns, at most
  static constexpr int RT = HB_PMAX / NL / 8;   // n8 tiles of R, at most
  static constexpr int KS = NL == 4 ? 2 : 1;
  static constexpr int RS = WPS / KS;
  static constexpr int MRB = HB_OMAX / NL / 16 / RS;  // 16-row blocks per warp, at most
  static constexpr int NACC = MRB * (NDT + 2 * RT) * 4;
  static constexpr int RED = KS > 1 ? NL * RS * (KS - 1) * 32 * NACC : 0;  // floats of the final reduction
};

template <bool LAST, bool SPLIT3, int NL>
__global__ void __launch_bounds__(THREADS) hss_wgrad_kernel(const __grid_constant__ HssWParams p) {
  using WL = WgradLayout<NL>;
  constexpr int WPS = WL::WPS, NDT = WL::NDT, RT = WL::RT, KS = WL::KS, RS = WL::RS, MRB = WL::MRB;
  static_assert(MRB >= 1 && RS * KS == WPS && HW_F % (8 * KS) == 0, "gradient layout");
  static_assert(RS > 1 || 2 * MRB <= NDT, "the dV A fragments come from the dD B fragments");
  extern __shared__ __align__(16) float smem[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int term = blockIdx.x / p.cpt, cta = blockIdx.x - term * p.cpt;
  if (term >= p.terms) return;
  const int stage = p.xtile + p.gtile + 2 * p.ptile;  // floats per pipeline stage
  const float* x0 = p.x + static_cast<long long>(term) * p.x_sz;
  const float* g0 = p.gr + static_cast<long long>(term) * p.g_sz;
  const float* p0 = p.pr + static_cast<long long>(term) * p.pj_sz;
  const float* q0 = p.qr + static_cast<long long>(term) * p.pj_sz;

  // Tile loads: per thread, at most LV 16-byte vectors of X and of G and LP of P
  // and of Q. Their offsets from the tile's first fiber are fixed (a tile lies in
  // one fiber block, checked by the host), so only that fiber's address changes.
  constexpr int LV = HW_F * HB_OMAX / 4 / THREADS;
  constexpr int LP = HW_F * HB_PMAX / 4 / THREADS;
  long long xo[LV], go[LV];
  int xd[LV], gd[LV], xf[LV], gf[LV], po[LP], pd[LP], pf[LP];
  auto setup = [&](int n, int pitch, const FastDiv& f_n4, long long ss, long long sk, long long (&o)[LV],
                   int (&d)[LV], int (&ff)[LV]) {
#pragma unroll
    for (int i = 0; i < LV; ++i) {
      const int v = tid + i * THREADS;
      ff[i] = HW_F;  // no vector
      if (v < HW_F * n / 4) {
        if constexpr (LAST) {  // [fiber][feature], vectors along the features
          const int f = fdiv(v, f_n4), k = (v - f * (n / 4)) * 4;
          o[i] = f * ss + k;
          d[i] = f * pitch + k;
          ff[i] = f;
        } else {  // [feature][fiber], vectors along the fibers
          const int k = v / (HW_F / 4), f = (v % (HW_F / 4)) * 4;
          o[i] = f + k * sk;
          d[i] = k * pitch + f;
          ff[i] = f;
        }
      }
    }
  };
  setup(p.I, p.xp, p.f_i4, p.x_s, p.x_sk, xo, xd, xf);
  setup(p.O, p.gp, p.f_o4, p.g_s, p.g_sk, go, gd, gf);
#pragma unroll
  for (int i = 0; i < LP; ++i) {  // projections: [fiber][nlr], contiguous in global memory
    const int v = tid + i * THREADS;
    pf[i] = HW_F;
    if (v < HW_F * p.nlr / 4) {
      const int f = fdiv(v, p.f_p4), c = (v - f * (p.nlr / 4)) * 4;
      po[i] = f * p.nlr + c;
      pd[i] = f * p.pq + c;
      pf[i] = f;
    }
  }
  // One tile (X, G, P, Q) into stage `buf`; the caller commits the group.
  auto load_tile = [&](int tile, int buf) {
    const int m0 = tile * HW_F;
    const float* xb = x0 + blocked(m0, p.x_blk, LAST ? p.x_s : 1, p.x_sb);
    const float* gb = g0 + blocked(m0, p.g_blk, LAST ? p.g_s : 1, p.g_sb);
    const float* pb = p0 + static_cast<long long>(m0) * p.nlr;
    const float* qb = q0 + static_cast<long long>(m0) * p.nlr;
    float* xs = smem + buf * stage;
    float* gs = xs + p.xtile;
    float* ps = gs + p.gtile;
    float* qs = ps + p.ptile;
#pragma unroll
    for (int i = 0; i < LV; ++i) {
      if (xf[i] < HW_F) {
        const bool ok = m0 + xf[i] < p.M;
        cp_async16(xs + xd[i], ok ? xb + xo[i] : x0, ok);
      }
      if (gf[i] < HW_F) {
        const bool ok = m0 + gf[i] < p.M;
        cp_async16(gs + gd[i], ok ? gb + go[i] : g0, ok);
      }
    }
#pragma unroll
    for (int i = 0; i < LP; ++i) {
      if (pf[i] < HW_F) {
        const bool ok = m0 + pf[i] < p.M;
        cp_async16(ps + pd[i], ok ? pb + po[i] : p0, ok);
        cp_async16(qs + pd[i], ok ? qb + po[i] : q0, ok);
      }
    }
  };
  // Element (fiber f, feature k) of an X or G tile.
  auto at = [&](const float* s, int pitch, int f, int k) { return LAST ? s[f * pitch + k] : s[k * pitch + f]; };

  auto split = [](float v, uint32_t& h, uint32_t& l) {
    h = to_tf32(v);
    if constexpr (SPLIT3) l = lo_part(v, h);
  };
  auto mma3 = [&](float (&d)[4], const uint32_t (&ah)[4], const uint32_t (&al)[4], const uint32_t (&bh)[2],
                  const uint32_t (&bl)[2]) {
    if constexpr (SPLIT3) {
      mma_tf32(d, al, bh);
      mma_tf32(d, ah, bl);
    }
    mma_tf32(d, ah, bh);
  };

  const int j = warp / WPS, sub = warp % WPS, ks = sub / RS, rs = sub % RS;
  const int rbd = p.ro / 16, rbv = p.ci / 16;  // 16-row blocks of dD and dU, of dV
  const int ndt = p.ci / 8, rt = p.R / 8;
  const int pg = j * p.R + (g & 3) * 2 + (g >> 2);  // column g of an 8-chunk of segment j's projections
  float aD[MRB][NDT][4], aU[MRB][RT][4], aV[MRB][RT][4], tD[MRB][NDT][4], tU[MRB][RT][4], tV[MRB][RT][4];
#pragma unroll
  for (int u = 0; u < MRB; ++u) {
#pragma unroll
    for (int n = 0; n < NDT; ++n)
#pragma unroll
      for (int q = 0; q < 4; ++q) aD[u][n][q] = tD[u][n][q] = 0.f;
#pragma unroll
    for (int n = 0; n < RT; ++n)
#pragma unroll
      for (int q = 0; q < 4; ++q) aU[u][n][q] = aV[u][n][q] = tU[u][n][q] = tV[u][n][q] = 0.f;
  }
  auto& dD = SPLIT3 ? tD : aD;
  auto& dU = SPLIT3 ? tU : aU;
  auto& dV = SPLIT3 ? tV : aV;

  auto compute_tile = [&](int buf) {
    const float* xs = smem + buf * stage;
    const float* gs = xs + p.xtile;
    const float* ps = gs + p.gtile;
    const float* qs = ps + p.ptile;
#pragma unroll
    for (int kv = 0; kv < HW_F / 8 / KS; ++kv) {
      const int k0 = (ks + kv * KS) * 8;
      uint32_t xh[NDT][2], xl[NDT][2], ph[RT][2], pl[RT][2], qh[RT][2], ql[RT][2];  // B, shared by the row blocks
#pragma unroll
      for (int n = 0; n < NDT; ++n)
        if (n < ndt) {  // X: fibers x the columns of leaf j
          split(at(xs, p.xp, k0 + t4, j * p.ci + n * 8 + g), xh[n][0], xl[n][0]);
          split(at(xs, p.xp, k0 + t4 + 4, j * p.ci + n * 8 + g), xh[n][1], xl[n][1]);
        }
#pragma unroll
      for (int n = 0; n < RT; ++n)
        if (n < rt) {  // P of output segment j, Q of input segment j
          split(ps[(k0 + t4) * p.pq + pg + n * 8], ph[n][0], pl[n][0]);
          split(ps[(k0 + t4 + 4) * p.pq + pg + n * 8], ph[n][1], pl[n][1]);
          split(qs[(k0 + t4) * p.pq + pg + n * 8], qh[n][0], ql[n][0]);
          split(qs[(k0 + t4 + 4) * p.pq + pg + n * 8], qh[n][1], ql[n][1]);
        }
#pragma unroll
      for (int u = 0; u < MRB; ++u) {
        const int rb = rs + u * RS;
        if (rb < rbd) {
          uint32_t ah[4], al[4];  // A = G^T: 16 rows o of leaf j x fibers
          const int o = j * p.ro + rb * 16 + g;
          split(at(gs, p.gp, k0 + t4, o), ah[0], al[0]);
          split(at(gs, p.gp, k0 + t4, o + 8), ah[1], al[1]);
          split(at(gs, p.gp, k0 + t4 + 4, o), ah[2], al[2]);
          split(at(gs, p.gp, k0 + t4 + 4, o + 8), ah[3], al[3]);
#pragma unroll
          for (int n = 0; n < NDT; ++n)
            if (n < ndt) mma3(dD[u][n], ah, al, xh[n], xl[n]);
#pragma unroll
          for (int n = 0; n < RT; ++n)
            if (n < rt) mma3(dU[u][n], ah, al, ph[n], pl[n]);
        }
        if (rb < rbv) {
          uint32_t ah[4], al[4];  // A = X^T: 16 columns i of leaf j x fibers
          if constexpr (RS == 1) {  // rb = u: the B fragments of columns 16 u .. 16 u + 15
            ah[0] = xh[2 * u][0];
            ah[1] = xh[2 * u + 1][0];
            ah[2] = xh[2 * u][1];
            ah[3] = xh[2 * u + 1][1];
            if constexpr (SPLIT3) {
              al[0] = xl[2 * u][0];
              al[1] = xl[2 * u + 1][0];
              al[2] = xl[2 * u][1];
              al[3] = xl[2 * u + 1][1];
            }
          } else {
            const int i = j * p.ci + rb * 16 + g;
            split(at(xs, p.xp, k0 + t4, i), ah[0], al[0]);
            split(at(xs, p.xp, k0 + t4, i + 8), ah[1], al[1]);
            split(at(xs, p.xp, k0 + t4 + 4, i), ah[2], al[2]);
            split(at(xs, p.xp, k0 + t4 + 4, i + 8), ah[3], al[3]);
          }
#pragma unroll
          for (int n = 0; n < RT; ++n)
            if (n < rt) mma3(dV[u][n], ah, al, qh[n], ql[n]);
        }
      }
    }
    if constexpr (SPLIT3) {
#pragma unroll
      for (int u = 0; u < MRB; ++u) {
#pragma unroll
        for (int n = 0; n < NDT; ++n)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            aD[u][n][q] += tD[u][n][q];
            tD[u][n][q] = 0.f;
          }
#pragma unroll
        for (int n = 0; n < RT; ++n)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            aU[u][n][q] += tU[u][n][q];
            aV[u][n][q] += tV[u][n][q];
            tU[u][n][q] = tV[u][n][q] = 0.f;
          }
      }
    }
  };

  // Pipeline: tile u of this CTA (fiber tile cta + u cpt) is in stage u % HW_STAGES,
  // and HW_STAGES - 1 tiles are loading while one computes.
  int tile = cta;
#pragma unroll
  for (int s = 0; s < HW_STAGES - 1; ++s) {
    if (tile + s * p.cpt < p.m_tiles) load_tile(tile + s * p.cpt, s);
    cp_async_commit();
  }
  for (int buf = 0; tile < p.m_tiles; tile += p.cpt) {
    cp_async_wait<HW_STAGES - 2>();
    __syncthreads();  // this tile is in; every warp is done with the stage the next load overwrites
    const int next = tile + (HW_STAGES - 1) * p.cpt;
    if (next < p.m_tiles) load_tile(next, buf == 0 ? HW_STAGES - 1 : buf - 1);
    cp_async_commit();
    compute_tile(buf);
    buf = buf == HW_STAGES - 1 ? 0 : buf + 1;
  }
  cp_async_wait<0>();

  // The fiber splits of a segment: ks > 0 hand their sums to ks = 0 through
  // shared memory (the tile stages are free now).
  if constexpr (KS > 1) {
    constexpr int NACC = WL::NACC;
    float* red = smem + ((j * RS + rs) * (KS - 1) * 32 + lane) * NACC;
    __syncthreads();
    if (ks > 0) {
      float* r = red + (ks - 1) * 32 * NACC;
#pragma unroll
      for (int u = 0; u < MRB; ++u) {
#pragma unroll
        for (int n = 0; n < NDT; ++n)
#pragma unroll
          for (int q = 0; q < 4; ++q) r[((u * NDT + n) * 4) + q] = aD[u][n][q];
#pragma unroll
        for (int n = 0; n < RT; ++n)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            r[MRB * NDT * 4 + (u * RT + n) * 4 + q] = aU[u][n][q];
            r[MRB * (NDT + RT) * 4 + (u * RT + n) * 4 + q] = aV[u][n][q];
          }
      }
    }
    __syncthreads();
    if (ks == 0) {
      for (int kx = 0; kx < KS - 1; ++kx) {
        const float* r = red + kx * 32 * NACC;
#pragma unroll
        for (int u = 0; u < MRB; ++u) {
#pragma unroll
          for (int n = 0; n < NDT; ++n)
#pragma unroll
            for (int q = 0; q < 4; ++q) aD[u][n][q] += r[((u * NDT + n) * 4) + q];
#pragma unroll
          for (int n = 0; n < RT; ++n)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              aU[u][n][q] += r[MRB * NDT * 4 + (u * RT + n) * 4 + q];
              aV[u][n][q] += r[MRB * (NDT + RT) * 4 + (u * RT + n) * 4 + q];
            }
        }
      }
    }
  }
  if (ks > 0) return;

  // This CTA's sums -> ws[term][cta], in the packed factor layout.
  float* out = p.ws + (static_cast<long long>(term) * p.cpt + cta) * p.fstride;
#pragma unroll
  for (int u = 0; u < MRB; ++u) {
    const int rb = rs + u * RS;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      if (rb < rbd) {
        const int o = j * p.ro + rb * 16 + g + hh * 8;
#pragma unroll
        for (int n = 0; n < NDT; ++n)
          if (n < ndt)
            *reinterpret_cast<float2*>(out + o * p.dp + n * 8 + 2 * t4) =
                make_float2(aD[u][n][2 * hh], aD[u][n][2 * hh + 1]);
#pragma unroll
        for (int n = 0; n < RT; ++n)
          if (n < rt)
            *reinterpret_cast<float2*>(out + (p.O + p.NL * p.R) * p.dp + o * p.ep + n * 8 + 2 * t4) =
                make_float2(aU[u][n][2 * hh], aU[u][n][2 * hh + 1]);
      }
      if (rb < rbv) {  // dV^T (columns i, c) -> the V^T rows of P: [j R + c][i']
        const int il = rb * 16 + g + hh * 8;
#pragma unroll
        for (int n = 0; n < RT; ++n)
          if (n < rt) {
            const int c = n * 8 + 2 * t4;
            out[(p.O + j * p.R + c) * p.dp + il] = aV[u][n][2 * hh];
            out[(p.O + j * p.R + c + 1) * p.dp + il] = aV[u][n][2 * hh + 1];
          }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Dense assembly of the HSS factors and its backward, one launch each.
//
// Every block of a dense factor is a dense leaf or a low-rank coupling u v^T
// (u: rows x rank, v: cols x rank). The host cuts each block into tasks of at
// most AT x AT outputs so that large blocks spread over many CTAs:
//   forward:  T_LEAF (copy a leaf tile), T_LOWRANK (one tile of u v^T);
//   backward: T_LEAF_GRAD (gradient tile -> leaf gradient),
//             T_GU (gu = G v for up to AT rows), T_GV (gv = G^T u for up to AT columns).
// The backward reductions run in a fixed order, so the results are deterministic.
constexpr int AT = 64;
constexpr int ATHREADS = 256;
constexpr int ARG = ATHREADS / AT;  // row groups of a tile
constexpr int RCHUNK = 8;           // ranks per pass of the backward reductions

enum TaskKind : int { T_LEAF = 0, T_LOWRANK = 1, T_LEAF_GRAD = 2, T_GU = 3, T_GV = 4 };

struct Task {
  const float* a;  // T_LEAF: leaf tile (row stride lda); T_LOWRANK: u rows; T_GU: v; T_GV: u
  const float* b;  // T_LOWRANK: v rows; T_LEAF_GRAD, T_GU, T_GV: dense gradient (row stride ld)
  float* out;      // T_LEAF, T_LOWRANK: dense factor (ld); T_LEAF_GRAD: leaf gradient (lda); T_GU, T_GV: gu, gv
  int rows, cols, rank, ld, lda, kind;
};

__global__ void __launch_bounds__(ATHREADS) assemble_kernel(const Task* __restrict__ tasks) {
  const Task t = tasks[blockIdx.x];
  const int tid = threadIdx.x;
  const int j = tid % AT, i0 = tid / AT;  // column of a tile, first of this thread's rows
  __shared__ float us[AT][AT], vs[AT][AT];  // T_LOWRANK: rank chunk of u^T and v^T
  __shared__ float red[ARG][RCHUNK][AT];    // T_GV: partial sums of each row group
  switch (t.kind) {
    case T_LEAF:
      if (j < t.cols)
        for (int i = i0; i < t.rows; i += ARG)
          t.out[static_cast<long long>(i) * t.ld + j] = t.a[static_cast<long long>(i) * t.lda + j];
      break;
    case T_LEAF_GRAD:
      if (j < t.cols)
        for (int i = i0; i < t.rows; i += ARG)
          t.out[static_cast<long long>(i) * t.lda + j] = t.b[static_cast<long long>(i) * t.ld + j];
      break;
    case T_LOWRANK: {  // out(i, j) = sum_k u(i, k) v(j, k), rows and cols <= AT
      float acc[AT / ARG];
#pragma unroll
      for (int s = 0; s < AT / ARG; ++s) acc[s] = 0.f;
      for (int k0 = 0; k0 < t.rank; k0 += AT) {
        const int kc = min(AT, t.rank - k0);
        __syncthreads();
        for (int e = tid; e < AT * kc; e += ATHREADS) {
          const int r = e / kc, k = e - r * kc;
          us[k][r] = r < t.rows ? t.a[static_cast<long long>(r) * t.rank + k0 + k] : 0.f;
          vs[k][r] = r < t.cols ? t.b[static_cast<long long>(r) * t.rank + k0 + k] : 0.f;
        }
        __syncthreads();
        for (int k = 0; k < kc; ++k) {
          const float vj = vs[k][j];
#pragma unroll
          for (int s = 0; s < AT / ARG; ++s) acc[s] = fmaf(us[k][i0 + s * ARG], vj, acc[s]);
        }
      }
      if (j < t.cols)
#pragma unroll
        for (int s = 0; s < AT / ARG; ++s) {
          const int i = i0 + s * ARG;
          if (i < t.rows) t.out[static_cast<long long>(i) * t.ld + j] = acc[s];
        }
      break;
    }
    case T_GU: {  // gu(i, k) = sum_j G(i, j) v(j, k): one warp per row, lanes over the columns
      const int warp = tid / 32, lane = tid % 32;
      for (int i = warp; i < t.rows; i += ATHREADS / 32) {
        const float* g = t.b + static_cast<long long>(i) * t.ld;
        for (int k0 = 0; k0 < t.rank; k0 += RCHUNK) {
          float acc[RCHUNK];
#pragma unroll
          for (int k = 0; k < RCHUNK; ++k) acc[k] = 0.f;
          for (int c = lane; c < t.cols; c += 32) {
            const float gc = g[c];
            const float* v = t.a + static_cast<long long>(c) * t.rank + k0;
#pragma unroll
            for (int k = 0; k < RCHUNK; ++k)
              if (k0 + k < t.rank) acc[k] = fmaf(gc, v[k], acc[k]);
          }
#pragma unroll
          for (int k = 0; k < RCHUNK; ++k) {
            float sum = acc[k];
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
            if (lane == 0 && k0 + k < t.rank) t.out[static_cast<long long>(i) * t.rank + k0 + k] = sum;
          }
        }
      }
      break;
    }
    case T_GV: {  // gv(j, k) = sum_i G(i, j) u(i, k): ARG row groups per column, then a fixed-order sum
      for (int k0 = 0; k0 < t.rank; k0 += RCHUNK) {
        float acc[RCHUNK];
#pragma unroll
        for (int k = 0; k < RCHUNK; ++k) acc[k] = 0.f;
        if (j < t.cols)
          for (int i = i0; i < t.rows; i += ARG) {
            const float gij = t.b[static_cast<long long>(i) * t.ld + j];
            const float* u = t.a + static_cast<long long>(i) * t.rank + k0;
#pragma unroll
            for (int k = 0; k < RCHUNK; ++k)
              if (k0 + k < t.rank) acc[k] = fmaf(gij, u[k], acc[k]);
          }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < RCHUNK; ++k) red[i0][k][j] = acc[k];
        __syncthreads();
        if (i0 == 0 && j < t.cols)
          for (int k = 0; k < RCHUNK && k0 + k < t.rank; ++k) {
            float sum = red[0][k][j];
            for (int r = 1; r < ARG; ++r) sum += red[r][k][j];
            t.out[static_cast<long long>(j) * t.rank + k0 + k] = sum;
          }
      }
      break;
    }
  }
}

// meta: (blocks, 8) int64 CPU tensor, one row {u_index, v_index (-1 for a leaf), axis, term,
// row0, col0, rows, cols} per block; params: the HSS parameter tensors; dense[axis]: the
// (terms, out, in) dense factors (forward) or their gradients (backward). The forward
// fills `dense` and returns nothing; the backward returns the gradients of `params`
// (views of one buffer, so a single allocation).
std::vector<torch::Tensor> assemble(std::vector<torch::Tensor> params, std::vector<torch::Tensor> dense,
                                    torch::Tensor meta, bool backward) {
  TORCH_CHECK(meta.device().is_cpu() && meta.scalar_type() == torch::kInt64 && meta.dim() == 2 &&
                  meta.size(1) == 8 && meta.is_contiguous(),
              "meta must be a contiguous (blocks, 8) int64 CPU tensor");
  const int64_t nblocks = meta.size(0);
  TORCH_CHECK(nblocks > 0 && !params.empty() && !dense.empty(), "nothing to assemble");
  const auto device = dense[0].device();
  const c10::cuda::CUDAGuard guard(device);
  for (const auto& d : dense)
    TORCH_CHECK(d.device() == device && d.scalar_type() == torch::kFloat32 && d.dim() == 3 && d.is_contiguous(),
                "dense factors must be contiguous (terms, out, in) float32 tensors on one device");
  std::vector<torch::Tensor> grads;
  if (backward) {
    int64_t total = 0;
    for (const auto& p : params) total += (p.numel() + 3) / 4 * 4;  // keep every gradient 16-byte aligned
    auto flat = torch::empty({total}, dense[0].options());
    int64_t off = 0;
    grads.reserve(params.size());
    for (const auto& p : params) {
      grads.push_back(flat.narrow(0, off, p.numel()).view(p.sizes()));
      off += (p.numel() + 3) / 4 * 4;
    }
  }
  std::vector<Task> tasks;
  tasks.reserve(static_cast<size_t>(nblocks) * 2);
  const int64_t* m = meta.data_ptr<int64_t>();
  const int64_t np = static_cast<int64_t>(params.size());
  for (int64_t bi = 0; bi < nblocks; ++bi, m += 8) {
    const int64_t ui = m[0], vi = m[1], axis = m[2], term = m[3], row0 = m[4], col0 = m[5], rows = m[6], cols = m[7];
    TORCH_CHECK(ui >= 0 && ui < np && vi < np && axis >= 0 && axis < static_cast<int64_t>(dense.size()),
                "bad block descriptor");
    const torch::Tensor& d = dense[axis];
    TORCH_CHECK(term >= 0 && term < d.size(0) && row0 >= 0 && col0 >= 0 && rows > 0 && cols > 0 &&
                    row0 + rows <= d.size(1) && col0 + cols <= d.size(2),
                "block out of range");
    const torch::Tensor& u = params[ui];
    TORCH_CHECK(u.device() == device && u.scalar_type() == torch::kFloat32 && u.is_contiguous(),
                "HSS parameters must be contiguous float32 tensors on the device of the factors");
    const int64_t ld = d.size(2);
    float* corner = d.data_ptr<float>() + (term * d.size(1) + row0) * ld + col0;
    Task t{};
    t.ld = static_cast<int>(ld);
    if (vi < 0) {  // dense leaf
      TORCH_CHECK(u.numel() == rows * cols, "leaf has the wrong size");
      t.lda = static_cast<int>(cols);
      t.kind = backward ? T_LEAF_GRAD : T_LEAF;
      for (int64_t r = 0; r < rows; r += AT)
        for (int64_t c = 0; c < cols; c += AT) {
          t.rows = static_cast<int>(std::min<int64_t>(AT, rows - r));
          t.cols = static_cast<int>(std::min<int64_t>(AT, cols - c));
          if (backward) {
            t.b = corner + r * ld + c;
            t.out = grads[ui].data_ptr<float>() + r * cols + c;
          } else {
            t.a = u.data_ptr<float>() + r * cols + c;
            t.out = corner + r * ld + c;
          }
          tasks.push_back(t);
        }
      continue;
    }
    const torch::Tensor& v = params[vi];
    TORCH_CHECK(v.device() == device && v.scalar_type() == torch::kFloat32 && v.is_contiguous() && u.dim() == 2 &&
                    v.dim() == 2 && u.size(0) == rows && v.size(0) == cols && u.size(1) == v.size(1) && u.size(1) > 0,
                "low-rank factors have the wrong shape");
    const int64_t rank = u.size(1);
    t.rank = static_cast<int>(rank);
    if (!backward) {
      t.kind = T_LOWRANK;
      for (int64_t r = 0; r < rows; r += AT)
        for (int64_t c = 0; c < cols; c += AT) {
          t.rows = static_cast<int>(std::min<int64_t>(AT, rows - r));
          t.cols = static_cast<int>(std::min<int64_t>(AT, cols - c));
          t.a = u.data_ptr<float>() + r * rank;
          t.b = v.data_ptr<float>() + c * rank;
          t.out = corner + r * ld + c;
          tasks.push_back(t);
        }
    } else {
      t.kind = T_GU;  // gu = G v, AT rows per task
      t.a = v.data_ptr<float>();
      t.cols = static_cast<int>(cols);
      for (int64_t r = 0; r < rows; r += AT) {
        t.rows = static_cast<int>(std::min<int64_t>(AT, rows - r));
        t.b = corner + r * ld;
        t.out = grads[ui].data_ptr<float>() + r * rank;
        tasks.push_back(t);
      }
      t.kind = T_GV;  // gv = G^T u, AT columns per task
      t.a = u.data_ptr<float>();
      t.rows = static_cast<int>(rows);
      for (int64_t c = 0; c < cols; c += AT) {
        t.cols = static_cast<int>(std::min<int64_t>(AT, cols - c));
        t.b = corner + c;
        t.out = grads[vi].data_ptr<float>() + c * rank;
        tasks.push_back(t);
      }
    }
  }
  TORCH_CHECK(tasks.size() < (size_t{1} << 31), "too many assembly tasks");
  const int64_t bytes = static_cast<int64_t>(tasks.size() * sizeof(Task));
  auto host = torch::empty({bytes}, torch::dtype(torch::kUInt8).pinned_memory(true));
  std::memcpy(host.data_ptr(), tasks.data(), static_cast<size_t>(bytes));
  auto dev = host.to(device, /*non_blocking=*/true);
  assemble_kernel<<<static_cast<unsigned>(tasks.size()), ATHREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const Task*>(dev.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grads;
}

template <bool A_KC, bool B_KC, int EPI, bool SPLIT3>
void launch(const Params& p, cudaStream_t stream) {
  constexpr int A_STAGE = A_KC ? BM * PITCH_K : BK * PITCH_MN;
  constexpr int B_STAGE = B_KC ? BN * PITCH_K : BK * PITCH_MN;
  constexpr size_t smem = static_cast<size_t>(STAGES) * (A_STAGE + B_STAGE) * sizeof(float);
  auto kernel = gemm_kernel<A_KC, B_KC, EPI, SPLIT3>;
  // Persistent grid: one wave of resident CTAs. The device queries cost tens of
  // microseconds, so they are cached per device (and per instantiation).
  constexpr int kMaxDevices = 64;
  static int resident_ctas[kMaxDevices] = {0};
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  TORCH_CHECK(device < kMaxDevices, "too many devices");
  if (resident_ctas[device] == 0) {
    int sms = 0, per_sm = 0;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, THREADS, smem));
    resident_ctas[device] = sms * std::max(per_sm, 1);
  }
  const int ctas = std::min(p.items, resident_ctas[device]);
  kernel<<<static_cast<unsigned>(ctas), THREADS, smem, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool A_KC, bool B_KC>
void dispatch_epi(const Params& p, int epi, bool split3, cudaStream_t stream) {
  if (epi == EPI_STORE) {
    split3 ? launch<A_KC, B_KC, EPI_STORE, true>(p, stream) : launch<A_KC, B_KC, EPI_STORE, false>(p, stream);
  } else {
    split3 ? launch<A_KC, B_KC, EPI_BIAS_ACT, true>(p, stream) : launch<A_KC, B_KC, EPI_BIAS_ACT, false>(p, stream);
  }
}

void check_tensor(const torch::Tensor& t, const char* name, int align_bytes) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32, name, " must be a float32 CUDA tensor");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % align_bytes == 0, name, " is not ", align_bytes,
              "-byte aligned");
}

// s = {sz, szb, s, sb, blk, sk, skb}. kc: K-contiguous (sk == 1), else the M/N index is (s == 1).
// Every other stride must keep 16-byte vectors aligned, and no vector may straddle a block.
Operand make_operand(const torch::Tensor& t, const std::vector<int64_t>& s, bool kc, int64_t extent, int64_t kblk,
                     const char* name) {
  TORCH_CHECK(s.size() == 7, name, ": expected 7 strides");
  Operand o{};
  o.ptr = t.data_ptr<float>();
  o.sz = s[0];
  o.szb = s[1];
  o.s = s[2];
  o.sb = s[3];
  o.sk = s[5];
  o.skb = s[6];
  o.blk = make_fastdiv(s[4]);
  for (int i : {0, 1, 2, 3, 5, 6}) TORCH_CHECK(s[i] >= 0, name, ": negative stride");
  if (kc) {
    TORCH_CHECK(o.sk == 1, name, ": K-contiguous operand needs a unit K stride");
    TORCH_CHECK(kblk % 4 == 0, name, ": K blocks of a K-contiguous operand must be multiples of 4");
    for (long long v : {o.sz, o.szb, o.s, o.sb, o.skb}) TORCH_CHECK(v % 4 == 0, name, ": strides must be multiples of 4");
  } else {
    TORCH_CHECK(o.s == 1, name, ": M/N-contiguous operand needs a unit M/N stride");
    TORCH_CHECK(extent % 4 == 0 && (s[4] >= extent || s[4] % 4 == 0), name,
                ": M/N extent and block must be multiples of 4");
    for (long long v : {o.sz, o.szb, o.sb, o.sk, o.skb}) TORCH_CHECK(v % 4 == 0, name, ": strides must be multiples of 4");
  }
  return o;
}

// Generic strided GEMM, strides in elements relative to each tensor's data pointer.
// shape = {M, N, K, kblk, batch, zblk, splits, tps}
// sa, sb = {sz, szb, s, sb, blk, sk, skb}    (see Operand)
// sc = {sz, szb, ssplit, sm, smb, mblk, snb, nblk}
// epi = EPI_BIAS_ACT: Y = acc + bias (optional) is written to c (optional) and
// LeakyReLU(Y) to c2 (optional, needs slope); bias is indexed by m, or by
// (n / bias_div) % bias_mod when bias_on_n.
void gemm(torch::Tensor a, torch::Tensor b, c10::optional<torch::Tensor> c, c10::optional<torch::Tensor> c2,
          c10::optional<torch::Tensor> bias, c10::optional<torch::Tensor> slope, std::vector<int64_t> shape,
          std::vector<int64_t> sa, std::vector<int64_t> sb, std::vector<int64_t> sc, bool a_kc, bool b_kc,
          int64_t epi, bool bias_on_n, int64_t bias_div, int64_t bias_mod, bool split3) {
  TORCH_CHECK(shape.size() == 8 && sc.size() == 8, "bad parameter lengths");
  check_tensor(a, "a", 16);
  check_tensor(b, "b", 16);
  const c10::cuda::CUDAGuard guard(a.device());
  const int64_t M = shape[0], N = shape[1], K = shape[2], kblk = shape[3], batch = shape[4], zblk = shape[5];
  const int64_t splits = shape[6], tps = shape[7];
  for (int64_t v : {M, N, K, kblk, batch, zblk}) TORCH_CHECK(v >= 1 && v <= BIG, "GEMM size out of range: ", v);
  TORCH_CHECK(K % kblk == 0, "K must be a multiple of the K block");
  const int64_t tpb = (kblk + BK - 1) / BK;
  const int64_t ktiles = K / kblk * tpb;
  TORCH_CHECK(ktiles < BIG, "too many K tiles");
  TORCH_CHECK(splits >= 1 && tps >= 1 && (splits - 1) * tps < ktiles && ktiles <= splits * tps,
              "every K split must be non-empty");
  const int64_t n_tiles = (N + BN - 1) / BN;
  const int64_t tiles = (M + BM - 1) / BM * n_tiles;
  const int64_t items = tiles * batch * splits;
  TORCH_CHECK(items < (int64_t{1} << 31) - 1, "too many work items");

  Params p{};
  p.M = static_cast<int>(M);
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);
  p.kblk = static_cast<int>(kblk);
  p.tpb = static_cast<int>(tpb);
  p.ktiles = static_cast<int>(ktiles);
  p.batch = static_cast<int>(batch);
  p.splits = static_cast<int>(splits);
  p.tps = static_cast<int>(tps);
  p.n_tiles = static_cast<int>(n_tiles);
  p.tiles = static_cast<int>(tiles);
  p.items = static_cast<int>(items);
  p.f_tpb = make_fastdiv(tpb);
  p.f_ntiles = make_fastdiv(n_tiles);
  p.f_tiles = make_fastdiv(tiles);
  p.f_splits = make_fastdiv(splits);
  p.f_zblk = make_fastdiv(zblk);
  p.a = make_operand(a, sa, a_kc, M, kblk, "A");
  p.b = make_operand(b, sb, b_kc, N, kblk, "B");

  TORCH_CHECK(N % 2 == 0 && (sc[7] >= N || sc[7] % 2 == 0), "C: N and its block must be even");
  for (int i : {0, 1, 2, 3, 4, 6}) TORCH_CHECK(sc[i] >= 0 && sc[i] % 2 == 0, "C: strides must be even");
  p.c_sz = sc[0];
  p.c_szb = sc[1];
  p.c_ssplit = sc[2];
  p.c_sm = sc[3];
  p.c_smb = sc[4];
  p.c_mblk = make_fastdiv(sc[5]);
  p.c_snb = sc[6];
  p.c_nblk = make_fastdiv(sc[7]);
  p.c = nullptr;
  if (c.has_value()) {
    check_tensor(*c, "c", 8);
    p.c = c->data_ptr<float>();
  }
  p.bias_on_n = bias_on_n ? 1 : 0;
  TORCH_CHECK(bias_div >= 1 && bias_div <= BIG && bias_mod >= 1 && bias_mod <= BIG, "bad bias indexing");
  p.bias_div = make_fastdiv(bias_div);
  p.bias_mod = make_fastdiv(bias_mod);
  if (epi == EPI_BIAS_ACT) {
    TORCH_CHECK(p.c != nullptr || c2.has_value(), "the bias/act epilogue needs c or c2");
    if (c2.has_value()) {
      TORCH_CHECK(slope.has_value(), "the activation needs a slope");
      check_tensor(*c2, "c2", 8);
      check_tensor(*slope, "slope", 4);
      p.c2 = c2->data_ptr<float>();
      p.slope = slope->data_ptr<float>();
    }
    if (bias.has_value()) {
      check_tensor(*bias, "bias", 4);
      TORCH_CHECK(bias->is_contiguous() && bias->numel() == (bias_on_n ? bias_mod : M), "bias has the wrong size");
      p.bias = bias->data_ptr<float>();
    }
  } else {
    TORCH_CHECK(epi == EPI_STORE, "unknown epilogue ", epi);
    TORCH_CHECK(p.c != nullptr, "c is required");
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (a_kc && b_kc) dispatch_epi<true, true>(p, static_cast<int>(epi), split3, stream);
  else if (a_kc && !b_kc) dispatch_epi<true, false>(p, static_cast<int>(epi), split3, stream);
  else if (!a_kc && b_kc) dispatch_epi<false, true>(p, static_cast<int>(epi), split3, stream);
  else dispatch_epi<false, false>(p, static_cast<int>(epi), split3, stream);
}

void act_backward(torch::Tensor dz, torch::Tensor y, torch::Tensor slope, torch::Tensor dy, torch::Tensor row_dbias,
                  torch::Tensor row_dslope, int64_t row_len) {
  for (auto* t : {&dz, &y, &dy}) check_tensor(*t, "tensor", 16);
  check_tensor(slope, "slope", 4);
  TORCH_CHECK(dz.is_contiguous() && y.is_contiguous() && dy.is_contiguous(), "act_backward needs contiguous tensors");
  TORCH_CHECK(dz.numel() == y.numel() && dy.numel() == y.numel(), "act_backward: size mismatch");
  TORCH_CHECK(row_len % 4 == 0 && row_len > 0 && dz.numel() % row_len == 0, "bad row length");
  const long long rows = dz.numel() / row_len;
  TORCH_CHECK(row_dbias.numel() == rows && row_dslope.numel() == rows, "act_backward: bad row buffers");
  TORCH_CHECK(rows < (1LL << 31) && row_len < (1LL << 31), "too many rows or a row too long");
  const c10::cuda::CUDAGuard guard(dz.device());
  if (rows == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // Rows of at most 64 float4 (two per lane): one warp each; longer rows: one block each.
  const bool long_rows = row_len / 4 > 64;
  const int64_t blocks = long_rows ? rows : (rows + AB_WARPS - 1) / AB_WARPS;
  auto kernel = long_rows ? act_backward_kernel<AB_WARPS> : act_backward_kernel<1>;
  kernel<<<static_cast<unsigned>(blocks), AB_WARPS * 32, 0, stream>>>(
      dz.data_ptr<float>(), y.data_ptr<float>(), slope.data_ptr<float>(), dy.data_ptr<float>(),
      row_dbias.data_ptr<float>(), row_dslope.data_ptr<float>(), static_cast<int>(row_len), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool LAST, int EPI, bool SPLIT3, int NL, int PJ>
void launch_hss(const HssParams& p, size_t smem, cudaStream_t stream) {
  auto kernel = hss_kernel<LAST, EPI, SPLIT3, NL, PJ>;
  constexpr int kMaxDevices = 64;
  static int optin[kMaxDevices] = {0}, sms[kMaxDevices] = {0}, ctas[kMaxDevices] = {0};
  static size_t ctas_smem[kMaxDevices] = {0};
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  TORCH_CHECK(device < kMaxDevices, "too many devices");
  if (optin[device] == 0) {
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin[device], cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms[device], cudaDevAttrMultiProcessorCount, device));
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin[device]));
  }
  TORCH_CHECK(smem <= static_cast<size_t>(optin[device]), "structured product: ", smem,
              " bytes of shared memory needed, ", optin[device], " available");
  if (ctas_smem[device] != smem) {  // persistent grid: one wave of resident CTAs
    int per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, THREADS, smem));
    ctas[device] = sms[device] * std::max(per_sm, 1);
    ctas_smem[device] = smem;
  }
  kernel<<<static_cast<unsigned>(std::min(p.items, ctas[device])), THREADS, smem, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// X tile pitch and size of a structured product (see hss_kernel).
int64_t hss_xpitch(bool last, int64_t I) { return last ? I + 4 : HB_F + 8; }
int64_t hss_xtile(bool last, int64_t I) { return (last ? HB_F : I) * hss_xpitch(last, I); }

size_t hss_smem_bytes(bool last, int64_t I, int64_t terms, int64_t fstride, int64_t pp) {
  return static_cast<size_t>(2 * hss_xtile(last, I) + terms * fstride + HB_F * pp) * sizeof(float);
}

// Structured mode product (see hss_kernel), strides in elements.
// shape = {M, O, I, terms, sum_terms, NL, L, r, R}
// sx = {x_sz, x_s, x_sb, x_blk, x_sk, x_skb}: X(z, m, t I + i) as in HssParams
// sc = {c_sz, c_sm, c_smb, c_mblk, c_sn}:     C(z, m, o) as in HssParams
// epi = EPI_BIAS_ACT: bias (optional) indexed by o, or by m when !bias_on_n, as
// (index / bias_div) % bias_mod; the result goes to c (optional) and LeakyReLU of it
// to c2 (optional, needs slope).
// proj (optional): (terms, M, NL R), receives the routed projections for the
// factor gradient (see HssParams::proj). With proj and no c or c2, only the
// projections are computed.
void hss_mode(torch::Tensor x, torch::Tensor fact, c10::optional<torch::Tensor> c, c10::optional<torch::Tensor> c2,
              c10::optional<torch::Tensor> bias, c10::optional<torch::Tensor> slope,
              c10::optional<torch::Tensor> proj, std::vector<int64_t> shape,
              std::vector<int64_t> sx, std::vector<int64_t> sc, bool last, int64_t epi, bool bias_on_n,
              int64_t bias_div, int64_t bias_mod, bool split3) {
  TORCH_CHECK(shape.size() == 9 && sx.size() == 6 && sc.size() == 5, "bad parameter lengths");
  check_tensor(x, "x", 16);
  check_tensor(fact, "fact", 16);
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t M = shape[0], O = shape[1], I = shape[2], terms = shape[3], sum_terms = shape[4];
  const int64_t NL = shape[5], L = shape[6], r = shape[7], R = shape[8];
  TORCH_CHECK(M >= 1 && M < BIG && terms >= 1 && L >= 1 && L <= 3 && r >= 1 && NL == (int64_t{1} << L),
              "structured product: bad sizes (levels must be 1, 2 or 3)");
  TORCH_CHECK(O <= HB_OMAX && O % (8 * NL) == 0 && I % (8 * NL) == 0 && I <= HB_OMAX,
              "structured product: needs O, I <= 128, multiples of 8 * 2^levels");
  TORCH_CHECK(R % 8 == 0 && L * r <= R && NL * R <= HB_PMAX, "structured product: bad rank padding");
  const int64_t ro = O / NL, ci = I / NL;
  const int64_t dp = ci + 4, ep = R + 4, pp = NL * R + 4;
  const int64_t fstride = (O * dp + NL * R * dp + O * ep + 3) / 4 * 4;
  TORCH_CHECK(fact.is_contiguous() && fact.numel() == terms * fstride, "packed factors have the wrong size");
  const int64_t m_tiles = (M + HB_F - 1) / HB_F;
  const int64_t items = sum_terms ? m_tiles : m_tiles * terms;
  TORCH_CHECK(items < (int64_t{1} << 31) - 1, "too many work items");
  for (int64_t v : sx) TORCH_CHECK(v >= 0, "X: negative stride");
  for (int64_t v : sc) TORCH_CHECK(v >= 0, "C: negative stride");
  TORCH_CHECK(sx[3] >= 1 && sx[3] <= BIG && sc[3] >= 1 && sc[3] <= BIG, "bad fiber blocks");
  if (last) {  // contiguous along the factor's axis
    TORCH_CHECK(sx[4] == 1 && sc[4] == 1, "X and C must be contiguous along the factor's axis");
    for (int i : {0, 1, 2, 5}) TORCH_CHECK(sx[i] % 4 == 0, "X: strides must be multiples of 4");
    for (int i : {0, 1, 2}) TORCH_CHECK(sc[i] % 2 == 0, "C: strides must be even");
  } else {  // contiguous along the fibers, 16-byte vectors of 4 fibers
    TORCH_CHECK(sx[1] == 1 && sc[1] == 1, "X and C must be contiguous along the fibers");
    TORCH_CHECK(M % 4 == 0 && (sx[3] >= M || sx[3] % 4 == 0), "X: fibers and their blocks must be multiples of 4");
    for (int i : {0, 2, 4, 5}) TORCH_CHECK(sx[i] % 4 == 0, "X: strides must be multiples of 4");
  }

  HssParams p{};
  p.M = static_cast<int>(M);
  p.O = static_cast<int>(O);
  p.I = static_cast<int>(I);
  p.terms = static_cast<int>(terms);
  p.sum_terms = sum_terms ? 1 : 0;
  p.NL = static_cast<int>(NL);
  p.L = static_cast<int>(L);
  p.r = static_cast<int>(r);
  p.R = static_cast<int>(R);
  p.ro = static_cast<int>(ro);
  p.ci = static_cast<int>(ci);
  p.dp = static_cast<int>(dp);
  p.ep = static_cast<int>(ep);
  p.pp = static_cast<int>(pp);
  p.xp = static_cast<int>(hss_xpitch(last, I));
  p.xtile = static_cast<int>(hss_xtile(last, I));
  p.fstride = static_cast<int>(fstride);
  p.m_tiles = static_cast<int>(m_tiles);
  p.items = static_cast<int>(items);
  p.f_mtiles = make_fastdiv(m_tiles);
  p.f_r = make_fastdiv(r);
  p.f_i4 = make_fastdiv(I / 4);
  p.x = x.data_ptr<float>();
  p.x_sz = sx[0];
  p.x_s = sx[1];
  p.x_sb = sx[2];
  p.x_blk = make_fastdiv(sx[3]);
  p.x_sk = sx[4];
  p.x_skb = sx[5];
  p.fact = fact.data_ptr<float>();
  p.c_sz = sc[0];
  p.c_sm = sc[1];
  p.c_smb = sc[2];
  p.c_mblk = make_fastdiv(sc[3]);
  p.c_sn = sc[4];
  p.bias_on_n = bias_on_n ? 1 : 0;
  TORCH_CHECK(bias_div >= 1 && bias_div <= BIG && bias_mod >= 1 && bias_mod <= BIG, "bad bias indexing");
  p.bias_div = make_fastdiv(bias_div);
  p.bias_mod = make_fastdiv(bias_mod);
  if (c.has_value()) {
    check_tensor(*c, "c", last ? 8 : 4);
    p.c = c->data_ptr<float>();
  }
  if (proj.has_value()) {
    check_tensor(*proj, "proj", 16);
    TORCH_CHECK(proj->is_contiguous() && proj->numel() == terms * M * NL * R, "proj has the wrong size");
    p.proj = proj->data_ptr<float>();
    p.pj_sz = M * NL * R;
  }
  const bool has_out = c.has_value() || c2.has_value();
  const int pj = p.proj == nullptr ? 0 : has_out ? 1 : 2;
  if (!has_out) {
    TORCH_CHECK(p.proj != nullptr && epi == EPI_STORE, "the structured product needs an output or proj");
  } else if (epi == EPI_BIAS_ACT) {
    TORCH_CHECK(p.c != nullptr || c2.has_value(), "the bias/act epilogue needs c or c2");
    if (c2.has_value()) {
      TORCH_CHECK(slope.has_value(), "the activation needs a slope");
      check_tensor(*c2, "c2", last ? 8 : 4);
      check_tensor(*slope, "slope", 4);
      p.c2 = c2->data_ptr<float>();
      p.slope = slope->data_ptr<float>();
    }
    if (bias.has_value()) {
      check_tensor(*bias, "bias", 4);
      TORCH_CHECK(bias->is_contiguous() && bias->numel() == bias_mod, "bias has the wrong size");
      p.bias = bias->data_ptr<float>();
    }
  } else {
    TORCH_CHECK(epi == EPI_STORE, "unknown epilogue ", epi);
    TORCH_CHECK(p.c != nullptr, "c is required");
  }
  const size_t smem = hss_smem_bytes(last, I, terms, fstride, pp);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool act = epi == EPI_BIAS_ACT;
#define HSS_NL(L_, E_, S_, P_)                                   \
  (NL == 2   ? launch_hss<L_, E_, S_, 2, P_>(p, smem, stream)    \
   : NL == 4 ? launch_hss<L_, E_, S_, 4, P_>(p, smem, stream)    \
             : launch_hss<L_, E_, S_, 8, P_>(p, smem, stream))
#define HSS_ACT(L_, S_) (pj == 0 ? HSS_NL(L_, EPI_BIAS_ACT, S_, 0) : HSS_NL(L_, EPI_BIAS_ACT, S_, 1))
#define HSS_STORE(L_, S_)                   \
  (pj == 0   ? HSS_NL(L_, EPI_STORE, S_, 0) \
   : pj == 1 ? HSS_NL(L_, EPI_STORE, S_, 1) \
             : HSS_NL(L_, EPI_STORE, S_, 2))
#define HSS_LAUNCH(L_) \
  (act ? (split3 ? HSS_ACT(L_, true) : HSS_ACT(L_, false)) : (split3 ? HSS_STORE(L_, true) : HSS_STORE(L_, false)))
  last ? HSS_LAUNCH(true) : HSS_LAUNCH(false);
#undef HSS_LAUNCH
#undef HSS_STORE
#undef HSS_ACT
#undef HSS_NL
}

// Shared memory of a structured product: lets the host decide whether it fits.
int64_t hss_smem(bool last, int64_t I, int64_t terms, int64_t fstride, int64_t pp) {
  return static_cast<int64_t>(hss_smem_bytes(last, I, terms, fstride, pp));
}


template <bool LAST, bool SPLIT3, int NL>
void launch_hss_wgrad(const HssWParams& p, int ctas, size_t smem, cudaStream_t stream) {
  auto kernel = hss_wgrad_kernel<LAST, SPLIT3, NL>;
  constexpr int kMaxDevices = 64;
  static int optin[kMaxDevices] = {0};
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  TORCH_CHECK(device < kMaxDevices, "too many devices");
  if (optin[device] == 0) {
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin[device], cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin[device]));
  }
  TORCH_CHECK(smem <= static_cast<size_t>(optin[device]), "structured gradient: ", smem,
              " bytes of shared memory needed, ", optin[device], " available");
  kernel<<<static_cast<unsigned>(ctas), THREADS, smem, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Floats of an X or G tile of the factor gradient, and of a projection tile.
int64_t hss_wtile(bool last, int64_t n) { return last ? HW_F * (n + 8) : n * (HW_F + 4); }
int64_t hss_wptile(int64_t nlr) { return HW_F * (nlr + 8); }

size_t hss_wgrad_smem_bytes(bool last, int64_t O, int64_t I, int64_t NL, int64_t R) {
  const int64_t stages = HW_STAGES * (hss_wtile(last, I) + hss_wtile(last, O) + 2 * hss_wptile(NL * R));
  const int64_t red = NL == 2 ? WgradLayout<2>::RED : NL == 4 ? WgradLayout<4>::RED : WgradLayout<8>::RED;
  return static_cast<size_t>(std::max(stages, red)) * sizeof(float);
}

// Structured factor gradient (see hss_wgrad_kernel), strides in elements.
// shape = {M, O, I, terms, NL, R}
// sx = {x_sz, x_s, x_sb, x_blk, x_sk}: X(t, m, i); sg: the same for G(t, m, o)
// p, q: (terms, M, NL R), the routed projections hss_mode saved for the product
// (forward packs) and for its input gradient (transposed packs).
// ws: (terms, cpt, fstride) float32, zero-initialised; receives each CTA's sums.
void hss_wgrad(torch::Tensor x, torch::Tensor gr, torch::Tensor pr, torch::Tensor qr, torch::Tensor ws,
               std::vector<int64_t> shape, std::vector<int64_t> sx, std::vector<int64_t> sg, bool last, bool split3) {
  TORCH_CHECK(shape.size() == 6 && sx.size() == 5 && sg.size() == 5, "bad parameter lengths");
  check_tensor(x, "x", 16);
  check_tensor(gr, "g", 16);
  check_tensor(pr, "p", 16);
  check_tensor(qr, "q", 16);
  check_tensor(ws, "ws", 16);
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t M = shape[0], O = shape[1], I = shape[2], terms = shape[3], NL = shape[4], R = shape[5];
  TORCH_CHECK(M >= 1 && M < BIG && terms >= 1 && (NL == 2 || NL == 4 || NL == 8),
              "structured gradient: bad sizes (2, 4 or 8 leaves)");
  TORCH_CHECK(O <= HB_OMAX && I <= HB_OMAX && O % (16 * NL) == 0 && I % (16 * NL) == 0,
              "structured gradient: needs O, I <= 128, multiples of 16 * 2^levels");
  TORCH_CHECK(R >= 8 && R % 8 == 0 && NL * R <= HB_PMAX, "structured gradient: bad rank padding");
  const int64_t ro = O / NL, ci = I / NL, nlr = NL * R;
  const int64_t dp = ci + 4, ep = R + 4;
  const int64_t fstride = (O * dp + NL * R * dp + O * ep + 3) / 4 * 4;
  for (const torch::Tensor* t : {&pr, &qr})
    TORCH_CHECK(t->is_contiguous() && t->numel() == terms * M * nlr, "projections have the wrong size");
  TORCH_CHECK(ws.is_contiguous() && ws.dim() == 3 && ws.size(0) == terms && ws.size(2) == fstride,
              "workspace has the wrong shape");
  const int64_t cpt = ws.size(1);
  TORCH_CHECK(cpt >= 1 && terms * cpt < (int64_t{1} << 31), "bad number of CTAs per term");
  for (int64_t v : sx) TORCH_CHECK(v >= 0, "X: negative stride");
  for (int64_t v : sg) TORCH_CHECK(v >= 0, "G: negative stride");
  TORCH_CHECK(sx[3] >= 1 && sx[3] <= BIG && sg[3] >= 1 && sg[3] <= BIG, "bad fiber blocks");
  for (int64_t blk : {sx[3], sg[3]})
    TORCH_CHECK(blk >= M || blk % HW_F == 0, "structured gradient: fiber blocks must be multiples of ", HW_F);
  for (const auto* s : {&sx, &sg}) {
    if (last) {
      TORCH_CHECK((*s)[4] == 1, "contiguous along the factor's axis expected");
      for (int i : {0, 1, 2}) TORCH_CHECK((*s)[i] % 4 == 0, "strides must be multiples of 4");
    } else {
      TORCH_CHECK((*s)[1] == 1 && M % 4 == 0 && ((*s)[3] >= M || (*s)[3] % 4 == 0),
                  "contiguous along the fibers expected, in blocks of 4");
      for (int i : {0, 2, 4}) TORCH_CHECK((*s)[i] % 4 == 0, "strides must be multiples of 4");
    }
  }

  HssWParams p{};
  p.M = static_cast<int>(M);
  p.O = static_cast<int>(O);
  p.I = static_cast<int>(I);
  p.terms = static_cast<int>(terms);
  p.NL = static_cast<int>(NL);
  p.R = static_cast<int>(R);
  p.ro = static_cast<int>(ro);
  p.ci = static_cast<int>(ci);
  p.nlr = static_cast<int>(nlr);
  p.dp = static_cast<int>(dp);
  p.ep = static_cast<int>(ep);
  p.fstride = static_cast<int>(fstride);
  p.xp = static_cast<int>(last ? I + 8 : HW_F + 4);
  p.gp = static_cast<int>(last ? O + 8 : HW_F + 4);
  p.pq = static_cast<int>(nlr + 8);
  p.xtile = static_cast<int>(hss_wtile(last, I));
  p.gtile = static_cast<int>(hss_wtile(last, O));
  p.ptile = static_cast<int>(hss_wptile(nlr));
  p.m_tiles = static_cast<int>((M + HW_F - 1) / HW_F);
  p.cpt = static_cast<int>(cpt);
  p.f_i4 = make_fastdiv(I / 4);
  p.f_o4 = make_fastdiv(O / 4);
  p.f_p4 = make_fastdiv(nlr / 4);
  p.x = x.data_ptr<float>();
  p.x_sz = sx[0];
  p.x_s = sx[1];
  p.x_sb = sx[2];
  p.x_blk = make_fastdiv(sx[3]);
  p.x_sk = sx[4];
  p.gr = gr.data_ptr<float>();
  p.g_sz = sg[0];
  p.g_s = sg[1];
  p.g_sb = sg[2];
  p.g_blk = make_fastdiv(sg[3]);
  p.g_sk = sg[4];
  p.pr = pr.data_ptr<float>();
  p.qr = qr.data_ptr<float>();
  p.pj_sz = M * nlr;
  p.ws = ws.data_ptr<float>();
  const size_t smem = hss_wgrad_smem_bytes(last, O, I, NL, R);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int ctas = static_cast<int>(terms * cpt);
#define HSSW_NL(L_, S_)                                                    \
  (NL == 2   ? launch_hss_wgrad<L_, S_, 2>(p, ctas, smem, stream)          \
   : NL == 4 ? launch_hss_wgrad<L_, S_, 4>(p, ctas, smem, stream)          \
             : launch_hss_wgrad<L_, S_, 8>(p, ctas, smem, stream))
  if (last) {
    split3 ? HSSW_NL(true, true) : HSSW_NL(true, false);
  } else {
    split3 ? HSSW_NL(false, true) : HSSW_NL(false, false);
  }
#undef HSSW_NL
}

int64_t hss_wgrad_smem(bool last, int64_t O, int64_t I, int64_t NL, int64_t R) {
  return static_cast<int64_t>(hss_wgrad_smem_bytes(last, O, I, NL, R));
}

}  // namespace hss

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hss_mode", &hss::hss_mode, "Structured (HSS) mode product on tensor cores");
  m.def("hss_smem", &hss::hss_smem, "Shared memory of a structured mode product");
  m.def("hss_wgrad", &hss::hss_wgrad, "Structured gradient of the HSS factors");
  m.def("hss_wgrad_smem", &hss::hss_wgrad_smem, "Shared memory of a structured factor gradient");
  m.def("gemm", &hss::gemm, "Strided TF32 / 3xTF32 tensor-core GEMM for HSS mode products");
  m.def("act_backward", &hss::act_backward, "Trainable LeakyReLU backward with bias/slope reductions");
  m.def("assemble", &hss::assemble,
        "Dense assembly of the HSS factors (forward), or the gradients of the HSS parameters (backward)");
}
