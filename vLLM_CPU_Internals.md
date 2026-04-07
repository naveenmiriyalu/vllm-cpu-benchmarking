# vLLM CPU Internals: Prefill, Decode, Parallelization & Memory

Deep dive into CPU implementation internals in vLLM.

## Table of Contents

- [Prefill vs Decode Handling](#prefill-vs-decode-handling)
- [Multi-Core Parallelization](#multi-core-parallelization)
- [Attention Kernel Implementations](#attention-kernel-implementations)
- [KV Cache Memory Management](#kv-cache-memory-management)
- [Code Structure](#code-structure)

---

## Prefill vs Decode Handling

### Architecture Overview

vLLM CPU uses **different strategies** for prefill and decode based on CPU architecture:

From `vllm/v1/attention/backends/cpu_attn.py:28`:
```python
_CPU_ARCH_PREFER_MIXED_BATCH = (CpuArchEnum.X86, CpuArchEnum.ARM, CpuArchEnum.S390X)
```

### Strategy 1: Mixed Batch (x86, ARM, S390X)

**All requests processed together using custom CPU attention kernels**

```python
# cpu_attn.py:113-118
if current_platform.get_cpu_architecture() not in _CPU_ARCH_PREFER_MIXED_BATCH:
    reorder_batch_threshold = 1
    self.use_sdpa_prefill = True
else:
    # Mixed batch - no separation
    self.use_sdpa_prefill = False
```

**Flow**:
```
Batch = [Prefill_Req1, Prefill_Req2, Decode_Req1, Decode_Req2]
         ↓
All processed together
         ↓
Custom CPU Attention Kernel (AVX512/AMX/NEON)
```

**Rationale**: x86/ARM/S390X have efficient SIMD instructions that can handle mixed workloads well.

### Strategy 2: Separated Batch (PowerPC, RISCV, others)

**Decode and prefill separated and use different kernels**

```python
# cpu_attn.py:157-171
if self.use_sdpa_prefill and causal:
    # Reorder: decode first, then prefill
    (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
        split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
            require_uniform=True,
        )
    )
    # Truncate metadata for decode-only processing
    num_reqs = num_decodes
    sdpa_start_loc = sdpa_start_loc[num_decodes:] - num_decode_tokens
```

**Flow**:
```
Original Batch = [Prefill_Req1, Decode_Req1, Prefill_Req2, Decode_Req2]
                  ↓
Reorder to:     [Decode_Req1, Decode_Req2, Prefill_Req1, Prefill_Req2]
                  ↓                          ↓
              Custom CPU Attn          PyTorch SDPA
              (with KV cache)          (prefill only)
```

**Why separate?**:
- **Decode**: Memory-bound, benefits from custom kernels with KV cache optimization
- **Prefill**: Compute-bound, PyTorch's SDPA (Scaled Dot Product Attention) is well-optimized

### Forward Pass Implementation

From `vllm/v1/attention/backends/cpu_attn.py:331-360`:

```python
def forward(self, query, key, value, kv_cache, attn_metadata, output):
    # 1. Write new KV to cache
    ops.cpu_attn_reshape_and_cache(key, value, key_cache, value_cache, ...)
    
    # 2. If using SDPA for prefill (separated strategy)
    if attn_metadata.use_sdpa_prefill:
        num_decode_tokens = attn_metadata.num_decode_tokens
        # Process prefill tokens with SDPA
        self._run_sdpa_forward(
            query[num_decode_tokens:],
            key[num_decode_tokens:],
            value[num_decode_tokens:],
            output[num_decode_tokens:], ...
        )
        num_actual_tokens = num_decode_tokens
    
    # 3. Process decode tokens (or all tokens in mixed mode)
    if num_actual_tokens > 0:
        ops.cpu_attention_with_kv_cache(
            query=query[:num_actual_tokens],
            key_cache=key_cache,
            value_cache=value_cache,
            output=output[:num_actual_tokens], ...
        )
```

---

## Multi-Core Parallelization

### Work Distribution Strategy

vLLM CPU uses **OpenMP parallel-for** to distribute attention work across cores.

#### Parallelization is at the **request-level**, NOT token-level!

From `csrc/cpu/cpu_attn_impl.hpp:88-128`:

```cpp
struct AttentionMetadata {
    int32_t workitem_group_num;       // Number of work items
    int32_t thread_num;               // omp_get_max_threads()
    int32_t effective_thread_num;     // Active threads
    AttentionWorkItemGroup* workitem_groups_ptr;  // Work items
    int32_t cu_workitem_num_per_thread[1025];     // Cumulative distribution
    
    AttentionMetadata(...) : thread_num(omp_get_max_threads()), ...
};

struct AttentionWorkItemGroup {
    int32_t req_id;                // Which request
    int32_t q_token_id_start;      // Start token in query
    int32_t q_token_num;           // Number of query tokens
    int32_t kv_split_pos_start;    // KV cache start
    int32_t kv_split_pos_end;      // KV cache end
    int64_t total_kv_len;          // Total KV length
    int32_t split_id;              // Split ID for long sequences
    int32_t local_split_id;        // Local split ID
};
```

### Work Scheduling

The **scheduler** (in C++) creates work items before OpenMP parallelization:

From `csrc/cpu/cpu_attn.cpp:63-65`:
```cpp
cpu_attention::AttentionScheduler scheduler;
torch::Tensor metadata = scheduler.schedule(input);
return metadata;  // Contains work distribution
```

### OpenMP Parallelization

From `csrc/cpu/cpu_attn_impl.hpp` (grepping showed multiple `#pragma omp`):

```cpp
// Example from cpu_attn_vec.hpp
#pragma omp parallel for collapse(2)
for (int thread_id = 0; thread_id < num_threads; thread_id++) {
    for (int work_id = start; work_id < end; work_id++) {
        AttentionWorkItemGroup& work_item = workitem_groups[work_id];
        
        // Each thread processes one or more work items
        // Work item = subset of tokens from one request
        process_attention(work_item.req_id,
                         work_item.q_token_id_start,
                         work_item.q_token_num,
                         work_item.kv_split_pos_start,
                         work_item.kv_split_pos_end);
    }
}
```

### How Work is Divided

**Question**: Is it batch-level or token-level parallelism?

**Answer**: **Both, hierarchically!**

```
Batch of Requests
 ├─ Request 1 (seq_len=512)
 │   ├─ WorkItem 1: tokens 0-255, KV 0-255      ← Thread 0
 │   └─ WorkItem 2: tokens 256-511, KV 256-511  ← Thread 1
 ├─ Request 2 (seq_len=128)
 │   └─ WorkItem 3: tokens 0-127, KV 0-127      ← Thread 2
 └─ Request 3 (seq_len=1024)
     ├─ WorkItem 4: tokens 0-255, KV 0-512      ← Thread 3
     ├─ WorkItem 5: tokens 256-511, KV 512-1024 ← Thread 4
     └─ ... (more splits)
```

**Strategy**:
1. **Long sequences** are split into multiple work items (KV splitting)
2. **Short sequences** get one work item per request
3. **OpenMP threads** grab work items dynamically (load balancing)

### Thread-Local Scratchpad Buffers

Each thread has its own scratchpad to avoid synchronization:

From `csrc/cpu/cpu_attn_impl.hpp:197-245`:
```cpp
class AttentionScratchPad {
    // Per-thread buffers (no sharing!)
    int8_t* thread_scratchpad_ptr;   // Thread-local workspace
    int8_t* reduction_scratchpad_ptr; // For KV-split reduction
    
    // Thread-local buffers:
    // - Q buffer: gathered query heads (for GQA)
    // - Q@K^T buffer: attention logits
    // - Partial outputs: intermediate results
    // - Max/Sum buffers: for softmax computation
};
```

**Memory layout per thread**:
```
Thread 0 Scratchpad:
├─ Q buffer          [q_tile_size × head_dim]
├─ Logits buffer     [max_num_q × k_tile_size]
├─ Output buffer     [q_tile_size × head_dim]
├─ Max buffer        [q_tile_size]
└─ Sum buffer        [q_tile_size]

Thread 1 Scratchpad: (same layout)
...
```

### KV Splitting for Long Sequences

When a sequence is very long, it's split across multiple work items:

```python
# From cpu_attn.py:183-184
input.enable_kv_split = True  # Enable splitting
```

**Example**:
```
Request with seq_len=2048, num_query_tokens=1 (decode)
├─ Split 1: Query token 0, attend to KV[0:1024]     → Thread A
└─ Split 2: Query token 0, attend to KV[1024:2048]  → Thread B

After both complete:
  Reduction step combines partial results using softmax math
```

From `csrc/cpu/cpu_attn_impl.hpp:59-85`:
```cpp
struct ReductionWorkItemGroup {
    int32_t req_id;
    int32_t q_token_id_start;
    int32_t q_token_id_num;
    int32_t split_start_id;  // Which splits to reduce
    int32_t split_num;       // How many splits
};
```

---

## Attention Kernel Implementations

### ISA (Instruction Set Architecture) Selection

vLLM CPU supports **5 different ISA backends**:

From `csrc/cpu/cpu_attn_impl.hpp:15`:
```cpp
enum class ISA { 
    AMX,    // Intel Advanced Matrix Extensions (Sapphire Rapids+)
    VEC,    // AVX512 (x86)
    VEC16,  // AVX2 or smaller head dimensions
    NEON,   // ARM NEON (AArch64)
    VXE     // IBM Z Vector Extensions (S390X)
};
```

### ISA Selection Logic

From `vllm/v1/attention/backends/cpu_attn.py:480-499`:

```python
def _get_attn_isa(dtype: torch.dtype, block_size: int, head_size: int) -> str:
    if head_size is not None and head_size % 32 != 0 and head_size % 16 == 0:
        return "vec16"  # Use narrower vectorization
    
    supports_amx = torch._C._cpu._is_amx_tile_supported()
    supports_arm = current_platform.get_cpu_architecture() == CpuArchEnum.ARM
    supports_vxe = current_platform.get_cpu_architecture() == CpuArchEnum.S390X
    
    if supports_amx and dtype in (torch.bfloat16,) and block_size % 32 == 0:
        return "amx"     # Intel AMX for bf16 with block_size=32
    elif block_size % 32 == 0:
        if supports_arm:
            return "neon"  # ARM NEON FMLA and BFMMLA
        elif supports_vxe:
            return "vxe"   # IBM Z
        else:
            return "vec"   # AVX512 on x86
    else:
        return "vec16"     # Fallback (AVX2)
```

**Decision tree**:
```
head_size % 16 == 0 but % 32 != 0?
├─ Yes → vec16 (narrow SIMD)
└─ No → Check hardware
    ├─ Intel + AMX + bf16 + block_size%32==0 → amx
    ├─ ARM + block_size%32==0 → neon
    ├─ S390X + block_size%32==0 → vxe
    ├─ x86 + block_size%32==0 → vec (AVX512)
    └─ Otherwise → vec16 (AVX2 fallback)
```

### AVX512 Implementation (VEC)

From `csrc/cpu/cpu_attn_vec.hpp`:

```cpp
#pragma omp parallel for collapse(2)
for (int thread_id = 0; thread_id < num_threads; thread_id++) {
    for (int work_id = start_work; work_id < end_work; work_id++) {
        // Use AVX512 intrinsics for:
        // 1. Q@K^T computation (matrix multiply)
        // 2. Softmax computation (vectorized exp, max, sum)
        // 3. Attention_weights @ V (another matrix multiply)
        
        // Pseudo-code for attention computation:
        for (kv_chunk in kv_cache) {
            // 1. Compute Q @ K^T with AVX512
            __m512 q_vec = _mm512_loadu_ps(query);
            __m512 k_vec = _mm512_loadu_ps(key_cache);
            __m512 qk = _mm512_mul_ps(q_vec, k_vec);
            // ... reduce across head_dim
            
            // 2. Softmax
            __m512 max_vec = _mm512_max_ps(qk, prev_max);
            __m512 exp_vec = _mm512_exp_ps(_mm512_sub_ps(qk, max_vec));
            // ... accumulate
            
            // 3. Weighted sum with V
            __m512 v_vec = _mm512_loadu_ps(value_cache);
            __m512 out = _mm512_fmadd_ps(attn_weight, v_vec, output);
        }
    }
}
```

**Key optimizations**:
- **Vectorized operations**: Process 16×fp32 or 32×fp16 per instruction
- **Fused multiply-add** (`FMA`): Combines multiply+add in one instruction
- **Cache-friendly access**: Blocked KV cache access patterns
- **Online softmax**: Compute softmax incrementally without storing all logits

### AMX Implementation (Intel Advanced Matrix Extensions)

From `csrc/cpu/cpu_attn_amx.hpp`:

```cpp
#pragma omp parallel for collapse(2)
for (...) {
    // Intel AMX uses tile registers (2D)
    // Tile configuration for bf16 matrix multiplication
    
    // 1. Configure tiles
    _tile_loadconfig(&tile_config);  // 8 tiles of 16×64 bf16 elements
    
    // 2. Load Q into tiles
    _tile_loadd(TMM0, query_ptr, stride);
    
    // 3. Load K from cache into tiles
    _tile_loadd(TMM1, key_cache_ptr, stride);
    
    // 4. Matrix multiply: TMM2 = TMM0 @ TMM1^T
    _tile_dpbf16ps(TMM2, TMM0, TMM1);  // BF16 matrix multiply
    
    // 5. Store results
    _tile_stored(TMM2, logits_ptr, stride);
    
    // ... softmax computation
    
    // 6. Attention @ V
    _tile_loadd(TMM3, value_cache_ptr, stride);
    _tile_dpbf16ps(TMM4, TMM2, TMM3);
    _tile_stored(TMM4, output_ptr, stride);
    
    // 7. Release tiles
    _tile_release();
}
```

**AMX advantages**:
- **Massive throughput**: 16×16×2 bf16 operations per cycle
- **Dedicated hardware**: Separate from AVX512 execution units
- **Lower power**: More efficient than AVX512 for matrix ops
- **Requirements**: Intel Sapphire Rapids or newer, bf16 dtype

### ARM NEON Implementation

From `csrc/cpu/cpu_attn_neon.hpp`:

```cpp
#pragma omp parallel for collapse(2)
for (...) {
    // ARM NEON intrinsics
    
    // BF16 support via BFMMLA (ARMv8.6+)
    if (supports_bfmmla) {
        // Use bf16 matrix multiply-accumulate
        bfloat16x8_t q_vec = vld1q_bf16(query);
        bfloat16x8_t k_vec = vld1q_bf16(key_cache);
        float32x4_t result = vbfmmlaq_f32(acc, q_vec, k_vec);
    } else {
        // Fallback to fp16 or fp32 FMLA
        float32x4_t q_vec = vld1q_f32(query);
        float32x4_t k_vec = vld1q_f32(key_cache);
        float32x4_t result = vfmaq_f32(acc, q_vec, k_vec);
    }
}
```

### Runtime Dispatch

All ISA variants are **compiled** but selected at **runtime**:

From `csrc/cpu/cpu_attn.cpp:10-23`:
```cpp
cpu_attention::ISA isa;
if (isa_hint == "amx") {
    isa = cpu_attention::ISA::AMX;
} else if (isa_hint == "vec") {
    isa = cpu_attention::ISA::VEC;
} else if (isa_hint == "vec16") {
    isa = cpu_attention::ISA::VEC16;
} else if (isa_hint == "neon") {
    isa = cpu_attention::ISA::NEON;
} else if (isa_hint == "vxe") {
    isa = cpu_attention::ISA::VXE;
}
```

Then template dispatch:
```cpp
CPU_ATTN_DISPATCH(head_dim, isa, [&]() {
    // This calls the appropriate template instantiation
    // e.g., AttentionImpl<ISA::AMX, bfloat16, 128>
    cpu_attention::AttentionMainLoop<attn_impl> mainloop;
    mainloop(&input);
});
```

### Build System

From `cmake/cpu_extension.cmake` and `setup.py`:

**AVX512 vs AVX2 variants**:
```python
if torch.cpu._is_avx512_supported():
    if torch.cpu._is_avx512_bf16_supported():
        import vllm._C  # Full AVX512 with BF16
    else:
        import vllm._C_AVX512  # AVX512 without BF16
else:
    import vllm._C_AVX2  # Fallback to AVX2
```

**Multiple binaries** are built:
- `_C.so`: AVX512 + BF16 (newest CPUs)
- `_C_AVX512.so`: AVX512 without BF16
- `_C_AVX2.so`: AVX2 only (older CPUs)

---

## KV Cache Memory Management

### Memory Allocation

From `vllm/v1/worker/gpu/attn_utils.py:88-102`:

```python
def _allocate_kv_cache(kv_cache_config: KVCacheConfig, device: torch.device):
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
    
    # Allocate raw int8 tensors (untyped memory blocks)
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        # Single large allocation per KV cache group
        tensor = torch.zeros(
            kv_cache_tensor.size,  # Total bytes
            dtype=torch.int8,      # Raw bytes
            device=device          # "cpu" for CPU backend
        )
        
        # Share same memory across layers
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor
    
    return kv_cache_raw_tensors
```

**Key points**:
1. **Single allocation** per KV cache group (not per layer!)
2. **Untyped memory**: Allocated as `int8` (raw bytes)
3. **CPU device**: `torch.zeros(..., device="cpu")`

### How Memory is Reserved from OS

**For CPU, `torch.zeros()` uses standard OS memory allocation:**

```
torch.zeros(size, dtype=torch.int8, device="cpu")
    ↓
PyTorch CPU Allocator
    ↓
C++ new[] or malloc()
    ↓
OS Memory (via mmap/sbrk)
    ↓
Physical RAM Pages
```

**Process**:
1. **Python call**: `torch.zeros(N * GB, dtype=int8, device="cpu")`
2. **PyTorch**: Requests N GB from CPU allocator
3. **C++ allocator**: Calls `posix_memalign()` or `aligned_alloc()`
4. **Kernel**: Maps virtual pages, initially zero (copy-on-write)
5. **Physical allocation**: Happens lazily on first write

**Important**: Memory is **not pinned** on CPU (unlike GPU):

From `vllm/platforms/cpu.py:447-448`:
```python
@classmethod
def is_pin_memory_available(cls) -> bool:
    return False  # CPU doesn't use pinned memory
```

### Memory Calculation

From `vllm/platforms/cpu.py:120-143`:

```python
@classmethod
def get_device_total_memory(cls, device_id: int = 0) -> int:
    from vllm.utils.mem_constants import GiB_bytes
    
    kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE
    
    if kv_cache_space is None:
        # Auto-calculate based on NUMA nodes
        node_dir = "/sys/devices/system/node"
        nodes = [d for d in os.listdir(node_dir) if d.startswith("node")]
        num_numa_nodes = len(nodes) or 1
        
        free_cpu_memory = psutil.virtual_memory().total // num_numa_nodes
        DEFAULT_CPU_MEM_UTILIZATION = 0.5
        kv_cache_space = int(free_cpu_memory * DEFAULT_CPU_MEM_UTILIZATION)
        
        logger.warning("VLLM_CPU_KVCACHE_SPACE not set. Using %s GiB for KV cache.",
                      format_gib(kv_cache_space))
    else:
        kv_cache_space *= GiB_bytes  # Convert GB to bytes
    
    return kv_cache_space
```

**Memory distribution**:
```
Total System RAM: 256 GB
├─ NUMA Node 0: 128 GB
│   └─ vLLM (if NUMA-bound): 128 GB × 50% = 64 GB KV cache
└─ NUMA Node 1: 128 GB
    └─ vLLM (if NUMA-bound): 128 GB × 50% = 64 GB KV cache

If VLLM_CPU_KVCACHE_SPACE=40:
    Explicit 40 GB allocation (overrides auto-calculation)
```

### Memory Layout in Blocks

From `vllm/v1/attention/backends/cpu_attn.py:67-74`:

```python
@staticmethod
def get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str = "auto",
) -> tuple[int, ...]:
    # CPU KV cache layout: [2, num_blocks, num_kv_heads, block_size, head_size]
    return 2, num_blocks, num_kv_heads, block_size, head_size
```

**Memory structure**:
```
KV Cache Tensor Shape: [2, num_blocks, num_kv_heads, block_size, head_size]
                        │   │           │             │           │
                        │   │           │             │           └─ e.g., 128
                        │   │           │             └─ e.g., 128 tokens/block
                        │   │           └─ e.g., 32 heads
                        │   └─ e.g., 10000 blocks
                        └─ 0=Key, 1=Value

Memory size = 2 × num_blocks × num_kv_heads × block_size × head_size × sizeof(dtype)
Example:     2 × 10000 × 32 × 128 × 128 × 2 bytes (bf16) = 20.97 GB
```

### Reshaping Raw Memory

From `vllm/v1/worker/gpu/attn_utils.py:105-144`:

```python
def _reshape_kv_cache(
    kv_cache_config: KVCacheConfig,
    kv_cache_raw_tensors: dict[str, torch.Tensor],
    attn_backends: dict[str, AttentionBackend],
) -> dict[str, torch.Tensor]:
    
    for layer_name in kv_cache_group_spec.layer_names:
        raw_tensor = kv_cache_raw_tensors[layer_name]
        
        # Calculate number of blocks that fit in allocated memory
        num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
        
        # Get backend-specific shape
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            num_blocks,
            kv_cache_spec.block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
        )
        # For CPU: (2, num_blocks, num_kv_heads, block_size, head_size)
        
        # Cast from int8 to actual dtype (bf16/fp16/fp32)
        raw_tensor = raw_tensor.view(dtype)
        
        # Reshape and permute to match backend expectations
        raw_tensor = raw_tensor.view(kv_cache_shape)
        kv_caches[layer_name] = raw_tensor.permute(*inv_order)
    
    return kv_caches
```

**Process**:
```
Raw allocation: torch.Tensor([X bytes], dtype=int8)
                ↓
View as dtype:  torch.Tensor([X/sizeof(dtype) elements], dtype=bf16)
                ↓
Reshape:        torch.Tensor([2, N, H, B, D], dtype=bf16)
                ↓
Permute:        Match backend stride order
                ↓
Final KV cache ready for attention kernels
```

### Block Management

From `vllm/v1/core/block_pool.py:129-181`:

```python
class BlockPool:
    def __init__(self, num_gpu_blocks: int, ...):
        # Note: "gpu" in variable names is historical,
        # actually means "device" (CPU or GPU)
        self.num_gpu_blocks = num_gpu_blocks
        
        # All blocks (fixed size pool)
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        
        # Free block queue (doubly-linked list)
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        
        # Cache for prefix caching (hash → block)
        self.cached_block_hash_to_block = BlockHashToBlockMap()
```

**Block allocation flow**:
```
Request arrives
    ↓
Scheduler requests blocks from BlockPool
    ↓
BlockPool.allocate() → pop from free_block_queue
    ↓
Block is mapped to request's slot_mapping
    ↓
Attention kernel writes to block via slot_mapping
    ↓
On request completion: block freed back to free_block_queue
```

### NUMA Awareness

From `vllm/platforms/cpu.py:194-199`:

```python
# Reserve at least one core for nixl_connector under p/d case
if vllm_config.kv_transfer_config and (
    envs.VLLM_CPU_NUM_OF_RESERVED_CPU == 0
    or envs.VLLM_CPU_NUM_OF_RESERVED_CPU is None
):
    os.environ["VLLM_CPU_NUM_OF_RESERVED_CPU"] = "1"
```

**For disaggregated prefill/decode with KV transfer**:
- Reserve 1+ cores for KV cache transfer operations
- Prevents frontend/transfer threads from competing with OpenMP threads

---

## Code Structure

### Key Files

**Python**:
```
vllm/
├── platforms/cpu.py                      # CPU platform configuration
├── v1/
│   ├── worker/
│   │   ├── cpu_worker.py                # CPU worker (thread binding)
│   │   └── cpu_model_runner.py          # CPU model runner (inherits GPU)
│   ├── attention/backends/
│   │   └── cpu_attn.py                  # CPU attention backend
│   └── core/
│       └── block_pool.py                # Block management (shared CPU/GPU)
└── envs.py                              # Environment variables
```

**C++**:
```
csrc/cpu/
├── cpu_attn.cpp                         # Entry points for attention
├── cpu_attn_impl.hpp                    # Attention implementation base
├── cpu_attn_amx.hpp                     # Intel AMX implementation
├── cpu_attn_vec.hpp                     # AVX512 implementation
├── cpu_attn_neon.hpp                    # ARM NEON implementation
├── cpu_attn_vxe.hpp                     # IBM Z implementation
├── cpu_attn_dispatch_generated.h        # Template dispatch (generated)
├── torch_bindings.cpp                   # PyTorch C++ bindings
└── utils.hpp                            # Utility functions
```

### Call Flow

```
Python: vllm serve (CPU mode)
    ↓
CPUWorker.init_device()
    ├─ Bind OpenMP threads to cores (cpu_worker.py:73-107)
    └─ Initialize distributed environment
    ↓
CPUModelRunner.load_model()
    └─ Load model weights to CPU
    ↓
GPUModelRunner.initialize_kv_cache()  # Shared with GPU
    ├─ Calculate num_blocks from VLLM_CPU_KVCACHE_SPACE
    ├─ Allocate raw tensors: torch.zeros(..., device="cpu")
    └─ Reshape to [2, num_blocks, num_kv_heads, block_size, head_size]
    ↓
Model forward pass
    ├─ CPUAttentionBackendImpl.forward()
    │   ├─ cpu_attn_reshape_and_cache() [C++]
    │   │   └─ Write new KV to cache blocks
    │   ├─ [Optional] SDPA for prefill (PyTorch)
    │   └─ cpu_attention_with_kv_cache() [C++]
    │       ├─ Schedule work items (C++ scheduler)
    │       ├─ #pragma omp parallel for
    │       └─ Per-thread attention computation
    │           ├─ AMX/AVX512/NEON intrinsics
    │           ├─ Q @ K^T (from cache)
    │           ├─ Softmax
    │           └─ Attn_weights @ V (from cache)
    └─ Return output tokens
```

### ISA Dispatch Mechanism

Template metaprogramming generates all ISA×dtype×head_dim combinations:

From `csrc/cpu/generate_cpu_attn_dispatch.py`:
```python
# Script generates cpu_attn_dispatch_generated.h
for isa in [AMX, VEC, VEC16, NEON, VXE]:
    for dtype in [fp16, bf16, fp32]:
        for head_dim in [32, 64, 80, 96, 112, 128, 160, 192, 224, 256]:
            generate_template_instantiation(isa, dtype, head_dim)
```

Runtime dispatch:
```cpp
#define CPU_ATTN_DISPATCH(HEAD_DIM, ISA, FUNC) \
    if (ISA == ISA::AMX && HEAD_DIM == 128) { \
        using attn_impl = AttentionImpl<ISA::AMX, scalar_t, 128>; \
        FUNC(); \
    } else if (ISA == ISA::VEC && HEAD_DIM == 128) { \
        using attn_impl = AttentionImpl<ISA::VEC, scalar_t, 128>; \
        FUNC(); \
    } ...
```

---

## Performance Characteristics

### Prefill Performance

**Compute-bound**:
- Limited by CPU FLOPS (GFLOPS)
- Benefits from:
  - High clock speeds
  - AMX/AVX512 for matrix operations
  - Large batch sizes (amortize overhead)
  
**Typical bottleneck**: Matrix multiplication (Q@K^T, Attn@V)

### Decode Performance

**Memory-bound**:
- Limited by memory bandwidth (GB/s)
- Each token needs to attend to **all** KV cache
- Small compute, large memory access

**Benefits from**:
- NUMA-local memory access
- Prefetching KV cache blocks
- Cache-friendly block access patterns

### Parallelization Efficiency

**Good cases**:
- Large batch with many requests
- Each request has similar length
- → Load balanced across threads

**Bad cases**:
- Small batch (few requests)
- Highly variable sequence lengths
- → Thread imbalance, some idle

---

## Summary

### Prefill/Decode Strategy

| Architecture | Strategy | Prefill Kernel | Decode Kernel |
|-------------|----------|----------------|---------------|
| x86, ARM, S390X | Mixed batch | Custom CPU | Custom CPU |
| PowerPC, RISCV | Separated | PyTorch SDPA | Custom CPU |

### Parallelization

- **Work unit**: Request-level + KV-split
- **Mechanism**: OpenMP parallel-for
- **Load balancing**: Dynamic work stealing
- **Scratchpads**: Thread-local buffers

### ISA Support

| ISA | CPU | Instructions | Dtype Priority |
|-----|-----|--------------|----------------|
| AMX | Intel Sapphire Rapids+ | Tile matrix ops | bf16 |
| VEC | x86 with AVX512 | 512-bit SIMD | fp16/bf16/fp32 |
| VEC16 | x86 with AVX2 | 256-bit SIMD | fp16/fp32 |
| NEON | ARM AArch64 | 128-bit SIMD, BFMMLA | bf16/fp16/fp32 |
| VXE | IBM Z | Vector extensions | bf16/fp16/fp32 |

### Memory Management

- **Allocation**: `torch.zeros(..., device="cpu")` → OS malloc
- **Size**: `VLLM_CPU_KVCACHE_SPACE` GB or auto (50% per NUMA node)
- **Layout**: `[2, num_blocks, num_kv_heads, block_size, head_size]`
- **NUMA**: Allocated on bound NUMA node for locality
- **Blocks**: Managed by BlockPool (same as GPU)
