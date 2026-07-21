struct ds4_metal_args_argsort {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
};

struct ds4_metal_args_argsort_merge {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
};

typedef void (argsort_t)(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Sort one float row into an index row. DS4 only exports the descending
// instance because router and indexer selection both need top-k order.
template<ds4_sort_order order>
kernel void kernel_argsort_f32_i32(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    // Stage this block's score slice in threadgroup memory (indices stay in
    // [i00, i00+ntg.x), so shmem_f32[idx - i00] replaces the device gather).
    // The host allocates ntg.x extra floats after the index array.  Values and
    // the comparison network are unchanged, so the permutation is identical.
    threadgroup float * shmem_f32 = (threadgroup float *) (shmem_i32 + ntg.x);
    if (i00 + col < args.ne00) {
        shmem_f32[col] = src0_row[i00 + col];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= args.ne00 ||
                       (shmem_i32[ixj] <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= args.ne00 ||
                       (shmem_i32[col] <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < args.ne0 && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shmem_i32[col];
    }
}

// Host-visible sort variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC>;

typedef void (argsort_merge_t)(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Merges sorted index runs produced by kernel_argsort_f32_i32. In the DS4 graph
// this finishes top-k over router or compressed-attention score rows.
template<ds4_sort_order order>
kernel void kernel_argsort_merge_f32_i32(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int im  = tgpig[0] / args.ne01;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    const int start = im * (2 * args.len);

    const int len0 = MIN(args.len, MAX(0, args.ne0 - (int)(start)));
    const int len1 = MIN(args.len, MAX(0, args.ne0 - (int)(start + args.len)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    device const int32_t * tmp1 = tmp0 + args.len;

    dst += start
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    device const float * src0_row = (device const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int k1 = MIN(MIN(k0 + chunk, total), args.top_k);

    if (k0 >= args.top_k) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == DS4_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == DS4_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

// Host-visible merge variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_merge_f32_i32_desc")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC>;

// DSpark fused Markov-bias argmax (Metal port of the CUDA
// dspark_markov_argmax_kernel). For one draft position it computes
// argmax_v(logits[v] + W2[v]·(d1*W1[prev])) over the vocab without reading
// the 129k-float logits row back to the CPU. W1 row and W2 are Q8_0
// (34-byte blocks: half scale + 32 int8). Two-stage reduction instead of the
// CUDA 64-bit atomicMax: stage 1 writes one packed key per threadgroup,
// stage 2 reduces the partial keys. The key packing is the CUDA one —
// monotonic float bits in the high word, ~index in the low word, so
// numeric max resolves ties to the smaller index.
struct ds4_metal_args_dspark_markov {
    uint32_t vocab;
    uint32_t rank_blocks;
    uint32_t n_part;
};

static inline ulong dspark_markov_pack_key(float v, uint i) {
    const uint f = as_type<uint>(v);
    const uint fkey = (f & 0x80000000u) ? ~f : (f | 0x80000000u);
    return ((ulong)fkey << 32) | (uint)(~i);
}

kernel void kernel_dspark_markov_argmax_part(
        constant ds4_metal_args_dspark_markov & args,
        device const float * logits,
        device const uchar * w1_row,
        device const uchar * w2,
        device       ulong * partials,
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  tgpg  [[threadgroups_per_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr uint NT = 256;
    threadgroup float state[256];
    threadgroup float vals[NT];
    threadgroup uint  idxs[NT];
    const uint tid = tiitg;

    if (tid < args.rank_blocks * 32u) {
        const uint b = tid >> 5, k = tid & 31u;
        device const uchar *blk = w1_row + (ulong)b * 34u;
        const float d = (float)(*(device const half *)blk);
        state[tid] = d * (float)((device const char *)(blk + 2))[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float best_v = -INFINITY;
    uint  best_i = 0u;
    for (uint i = tgpig.x * NT + tid; i < args.vocab; i += tgpg.x * NT) {
        device const uchar *row = w2 + (ulong)i * args.rank_blocks * 34u;
        float acc = 0.0f;
        for (uint b = 0; b < args.rank_blocks; b++) {
            device const uchar *blk = row + (ulong)b * 34u;
            const float d = (float)(*(device const half *)blk);
            /* 34-byte Q8_0 blocks leave the quants 2 mod 4: packed_char4
             * (1-byte alignment) is required — a char4 load here is UB. */
            device const packed_char4 *q4 = (device const packed_char4 *)(blk + 2);
            float4 s4 = float4(0.0f);
            FOR_UNROLL (uint k = 0; k < 8u; k++) {
                s4 += float4(char4(q4[k])) * float4(state[b*32u + k*4u + 0u],
                                                    state[b*32u + k*4u + 1u],
                                                    state[b*32u + k*4u + 2u],
                                                    state[b*32u + k*4u + 3u]);
            }
            acc += d * (s4.x + s4.y + s4.z + s4.w);
        }
        const float v = logits[i] + acc;
        if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
    }

    vals[tid] = best_v;
    idxs[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = NT/2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float ov = vals[tid + stride];
            const uint  oi = idxs[tid + stride];
            if (ov > vals[tid] || (ov == vals[tid] && oi < idxs[tid])) {
                vals[tid] = ov;
                idxs[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) partials[tgpig.x] = dspark_markov_pack_key(vals[0], idxs[0]);
}

kernel void kernel_dspark_markov_argmax_final(
        constant ds4_metal_args_dspark_markov & args,
        device const ulong * partials,
        device       ulong * out_key,
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr uint NT = 128;
    threadgroup ulong keys[NT];
    const uint tid = tiitg;
    ulong best = 0;
    for (uint i = tid; i < args.n_part; i += NT) best = max(best, partials[i]);
    keys[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = NT/2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) keys[tid] = max(keys[tid], keys[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out_key[0] = keys[0];
}
