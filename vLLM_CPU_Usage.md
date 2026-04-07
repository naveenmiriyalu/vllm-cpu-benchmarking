# vLLM CPU Configuration Guide

Comprehensive guide for configuring vLLM on CPU with proper NUMA binding and core affinity.

## Table of Contents

- [CPU-Specific Environment Variables](#cpu-specific-environment-variables)
- [Understanding CPU Binding Behavior](#understanding-cpu-binding-behavior)
- [NUMA Topology and Memory Binding](#numa-topology-and-memory-binding)
- [Container Deployment](#container-deployment)
- [Common Configuration Patterns](#common-configuration-patterns)
- [Verification and Monitoring](#verification-and-monitoring)
- [Troubleshooting](#troubleshooting)

---

## CPU-Specific Environment Variables

### Core Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_CPU_KVCACHE_SPACE` | `0` (→ 4GB) | KV cache space in GB |
| `VLLM_CPU_OMP_THREADS_BIND` | `"auto"` | CPU cores for OpenMP threads |
| `VLLM_CPU_NUM_OF_RESERVED_CPU` | `None` | Number of cores to reserve (only with `auto` binding) |
| `VLLM_CPU_SGL_KERNEL` | `False` | Use small-batch optimized kernels (x86 only) |
| `CPU_VISIBLE_MEMORY_NODES` | - | Visible NUMA nodes (only with `auto` binding) |

### Location in Code

- **Environment definitions**: `vllm/vllm/envs.py`
- **Platform implementation**: `vllm/vllm/platforms/cpu.py`
- **Worker implementation**: `vllm/vllm/v1/worker/cpu_worker.py`
- **Model runner**: `vllm/vllm/v1/worker/cpu_model_runner.py`
- **Attention backend**: `vllm/vllm/v1/attention/backends/cpu_attn.py`

---

## Understanding CPU Binding Behavior

### How VLLM_CPU_OMP_THREADS_BIND Works

This variable controls which CPU cores OpenMP threads bind to. It supports multiple formats:

#### Supported Formats

```bash
# Range notation
export VLLM_CPU_OMP_THREADS_BIND=0-31
export VLLM_CPU_OMP_THREADS_BIND=65-66

# Comma-separated list
export VLLM_CPU_OMP_THREADS_BIND=0,1,2,65,66,67

# Mixed format
export VLLM_CPU_OMP_THREADS_BIND=0-31,33,64-95

# Multiple ranks (tensor parallel) - separated by |
export VLLM_CPU_OMP_THREADS_BIND=0-31|32-63

# Special values
export VLLM_CPU_OMP_THREADS_BIND=auto    # Auto-detect based on NUMA
export VLLM_CPU_OMP_THREADS_BIND=nobind  # No binding
```

### Important Behavior Notes

#### 1. Manual Binding vs Auto Binding

**Manual binding** (specific cores like `"0-1"` or `"65,67"`):
- ✅ Explicit control over core assignment
- ❌ `VLLM_CPU_NUM_OF_RESERVED_CPU` is **IGNORED**
- ❌ `CPU_VISIBLE_MEMORY_NODES` is **IGNORED**
- ⚠️ You must manually ensure cores match NUMA node

**Auto binding** (`"auto"`):
- ✅ Automatically selects cores from appropriate NUMA node
- ✅ `VLLM_CPU_NUM_OF_RESERVED_CPU` **WORKS**
- ✅ `CPU_VISIBLE_MEMORY_NODES` **WORKS**
- ✅ NUMA-aware by default

#### 2. Core Reservation Logic (Auto Mode Only)

From `vllm/v1/worker/cpu_worker.py:210-223`:

```python
reserve_cpu_num = envs.VLLM_CPU_NUM_OF_RESERVED_CPU
if reserve_cpu_num is None:
    need_reserve = (world_size > 1 or data_parallel_size_local > 1)
    reserve_cpu_num = 1 if need_reserve else 0

# Removes LAST N cores from the binding list
if reserve_cpu_num != 0:
    logical_cpu_list = logical_cpu_list[:-reserve_cpu_num]
```

**Key points**:
- Reserves **last** cores in the sorted list
- Only works when `VLLM_CPU_OMP_THREADS_BIND=auto`
- Default: reserves 1 core if world_size > 1, otherwise 0

#### 3. CPU_VISIBLE_MEMORY_NODES (Auto Mode Only)

From `vllm/platforms/cpu.py:383-388`:

```python
if env_key in os.environ and os.environ[env_key] != "":
    visible_nodes = [int(s) for s in os.environ[env_key].split(",")]
    allowed_numa_nodes_list = [
        x for x in sorted(list(set(visible_nodes))) if x in allowed_numa_nodes
    ]
```

Controls which NUMA nodes vLLM can use for auto binding:

```bash
# Only use NUMA nodes 0 and 2
export CPU_VISIBLE_MEMORY_NODES=0,2
export VLLM_CPU_OMP_THREADS_BIND=auto
```

---

## NUMA Topology and Memory Binding

### Understanding NUMA Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  NUMA Node 0    │         │  NUMA Node 1    │
│                 │         │                 │
│  Cores: 0-63    │         │  Cores: 64-127  │
│  Local Memory   │◄───────►│  Local Memory   │
│  (Fast access)  │  (QPI)  │  (Fast access)  │
└─────────────────┘         └─────────────────┘
```

### Find Your NUMA Topology

```bash
# View NUMA topology
numactl --hardware

# Or use lscpu
lscpu | grep -A 30 "NUMA"

# Check which NUMA node owns a specific core
cat /sys/devices/system/cpu/cpu65/topology/physical_package_id

# List all cores on NUMA node 1
cat /sys/devices/system/node/node1/cpulist
```

### The NUMA Mismatch Problem

**Problem**: Cores on one NUMA node accessing memory on another NUMA node

```bash
# BAD: Cross-NUMA access
numactl --cpunodebind=1 --membind=1 podman run ...
# Inside container:
export VLLM_CPU_OMP_THREADS_BIND=0-1  # Cores on node 0!
```

**Result**: Cores 0-1 (NUMA node 0) accessing memory on NUMA node 1
- ❌ Higher latency
- ❌ Lower bandwidth
- ❌ Poor performance

**Solution**: Match cores to memory NUMA node

```bash
# GOOD: Local NUMA access
numactl --cpunodebind=1 --membind=1 podman run ...
# Inside container:
export VLLM_CPU_OMP_THREADS_BIND=65-66  # Cores on node 1!
```

### Binding Precedence (Weakest → Strongest)

```
1. numactl --cpunodebind=N         ← Soft preference (can be overridden)
2. cgroup cpuset                   ← Container limit
3. taskset -c X,Y,Z                ← Hard affinity mask
4. sched_setaffinity() in code     ← Explicit binding (STRONGEST)
   ↑ This is what VLLM_CPU_OMP_THREADS_BIND uses
```

**Important**: vLLM's explicit `VLLM_CPU_OMP_THREADS_BIND` **overrides** numactl's `--cpunodebind`!

---

## Container Deployment

### Podman/Docker with numactl and taskset

#### Typical Launch Pattern

```bash
taskset -c $SERVER_CPULIST \
  numactl --cpunodebind=$SERVER_NUMA_NODE --membind=$SERVER_NUMA_NODE \
    podman run \
      -e VLLM_CPU_OMP_THREADS_BIND=$CORES \
      -e VLLM_CPU_KVCACHE_SPACE=$CACHE_GB \
      <image> vllm serve <model>
```

#### What Each Layer Does

```
┌─────────────────────────────────────────────────────┐
│ taskset -c 65,67                                    │  ← Hard CPU mask
│ ├── numactl --membind=1                            │  ← Memory on node 1
│ └── podman run                                      │
│     └── Container                                   │
│         ├── vLLM frontend (can use 65 or 67)       │
│         └── OpenMP threads (bind per env var)      │
│             ├── VLLM_CPU_OMP_THREADS_BIND=65       │
│             └── Threads pinned to core 65          │
└─────────────────────────────────────────────────────┘
```

### Podman-Specific Options

```bash
# Option 1: Use --cpuset-cpus
podman run \
  --cpuset-cpus=65,67 \
  --cpuset-mems=1 \
  -e VLLM_CPU_OMP_THREADS_BIND=65 \
  <image> vllm serve <model>

# Option 2: Combine with host taskset
taskset -c 65,67 podman run \
  -e VLLM_CPU_OMP_THREADS_BIND=65 \
  <image> vllm serve <model>
```

### Complete Working Example

```bash
# 1. Find NUMA node for target cores
NUMA_NODE=$(cat /sys/devices/system/cpu/cpu65/topology/physical_package_id)
echo "Core 65 is on NUMA node: $NUMA_NODE"

# 2. Get all cores on that NUMA node
CORES_ON_NODE=$(cat /sys/devices/system/node/node${NUMA_NODE}/cpulist)
echo "Cores on NUMA node $NUMA_NODE: $CORES_ON_NODE"

# 3. Launch with proper binding
taskset -c 65,67 \
  numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} \
  podman run -d \
    --name vllm-server \
    -e VLLM_CPU_OMP_THREADS_BIND=65 \
    -e VLLM_CPU_KVCACHE_SPACE=40 \
    -p 8000:8000 \
    vllm/vllm-cpu:latest \
    vllm serve facebook/opt-125m --dtype=bfloat16

# Result:
# - Core 65: OpenMP inference threads (high utilization)
# - Core 67: Frontend/asyncio (low utilization)
# - Memory: On NUMA node matching cores 65,67
```

---

## Common Configuration Patterns

### Pattern 1: Single Core for Everything

```bash
# Simplest setup - everything on one core
export VLLM_CPU_OMP_THREADS_BIND=65
export VLLM_CPU_KVCACHE_SPACE=40

# With container
taskset -c 65 \
  numactl --cpunodebind=1 --membind=1 \
  podman run -e VLLM_CPU_OMP_THREADS_BIND=65 <image> vllm serve <model>
```

**Use case**: Benchmarking, isolation, testing

### Pattern 2: Separate Frontend and Inference

```bash
# Two cores: one for inference, one for frontend
export VLLM_CPU_OMP_THREADS_BIND=65  # OpenMP only on 65
export VLLM_CPU_KVCACHE_SPACE=40

# With container - allow both cores, but OMP only uses 65
taskset -c 65,67 \
  numactl --cpunodebind=1 --membind=1 \
  podman run -e VLLM_CPU_OMP_THREADS_BIND=65 <image> vllm serve <model>
```

**Use case**: Production serving, better responsiveness

### Pattern 3: Auto Binding with Reservation

```bash
# Let vLLM auto-discover and reserve cores
export VLLM_CPU_OMP_THREADS_BIND=auto
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
export CPU_VISIBLE_MEMORY_NODES=1
export VLLM_CPU_KVCACHE_SPACE=40

# With container - restrict to NUMA node 1
NUMA1_CORES=$(cat /sys/devices/system/node/node1/cpulist)
taskset -c $NUMA1_CORES \
  numactl --cpunodebind=1 --membind=1 \
  podman run \
    -e VLLM_CPU_OMP_THREADS_BIND=auto \
    -e VLLM_CPU_NUM_OF_RESERVED_CPU=1 \
    -e CPU_VISIBLE_MEMORY_NODES=1 \
    <image> vllm serve <model>
```

**Use case**: Multi-worker, automatic NUMA-aware binding

### Pattern 4: Tensor Parallel (Multiple Workers)

```bash
# 2 workers, each on different NUMA node
export VLLM_CPU_OMP_THREADS_BIND=0-31|32-63  # Worker 0 | Worker 1
export VLLM_CPU_KVCACHE_SPACE=40

# Or with auto binding
export VLLM_CPU_OMP_THREADS_BIND=auto
export VLLM_CPU_NUM_OF_RESERVED_CPU=1

vllm serve <model> --tensor-parallel-size 2
```

**Use case**: Large models requiring multiple workers

### Pattern 5: Disaggregated Prefill/Decode (with KV Transfer)

```bash
# Prefill instance
export VLLM_CPU_OMP_THREADS_BIND=auto
export CPU_VISIBLE_MEMORY_NODES=0
export VLLM_CPU_NUM_OF_RESERVED_CPU=1  # Reserve for KV transfer

# Decode instance
export VLLM_CPU_OMP_THREADS_BIND=auto
export CPU_VISIBLE_MEMORY_NODES=1
export VLLM_CPU_NUM_OF_RESERVED_CPU=1  # Reserve for KV transfer
```

**Use case**: Prefill/decode disaggregation

---

## Verification and Monitoring

### Check Container CPU Affinity

```bash
# Get container PID
CPID=$(podman inspect --format '{{.State.Pid}}' <container-name>)

# Check taskset affinity
taskset -cp $CPID
# Output: current affinity list: 65,67

# Check cgroup cpuset
cat /proc/$CPID/status | grep Cpus_allowed_list
# Output: Cpus_allowed_list: 65,67

# Check which cores threads are actually running on
ps -eLo pid,tid,psr,comm -p $CPID | grep -E "(python|vllm)"
# PSR column shows current CPU core
```

### Check NUMA Binding

```bash
# Check memory binding
cat /proc/$CPID/numa_maps | grep bind

# Check NUMA statistics
numastat -p $CPID

# Expected for good configuration:
# - High "numa_hit" count (local access)
# - Low "numa_miss" or "numa_foreign" (cross-node access)
```

### Real-time CPU Monitoring

```bash
# 1. mpstat - Per-core utilization
mpstat -P 65,67 1  # Update every 1 second

# 2. htop - Interactive
htop
# Press F2 → Setup → Display options → "Detailed CPU time"

# 3. perf - Hardware cycles
sudo perf stat -C 65,67 -e cycles,instructions -I 1000

# 4. pidstat - Per-process
pidstat -p $CPID -u -t 1  # Shows which CPU each thread uses

# 5. turbostat - Detailed CPU metrics (Intel/AMD)
sudo turbostat --cpu 65,67 --interval 1
```

### Inside Container Verification

```bash
# Enter running container
podman exec -it <container-name> bash

# Check visible CPUs
cat /sys/fs/cgroup/cpuset/cpuset.cpus

# Check OpenMP settings
echo $OMP_NUM_THREADS
cat /proc/self/status | grep Cpus_allowed_list

# Check vLLM process affinity
ps -eLo pid,tid,psr,comm | grep vllm
```

---

## Troubleshooting

### Issue 1: Both Cores Active Despite RESERVED_CPU=1

**Symptoms**: Set `VLLM_CPU_OMP_THREADS_BIND=65,67` and `VLLM_CPU_NUM_OF_RESERVED_CPU=1`, but both cores show high utilization.

**Cause**: `VLLM_CPU_NUM_OF_RESERVED_CPU` only works with `VLLM_CPU_OMP_THREADS_BIND=auto`

**Solution**:
```bash
# Option A: Use auto binding
export VLLM_CPU_OMP_THREADS_BIND=auto
export VLLM_CPU_NUM_OF_RESERVED_CPU=1

# Option B: Manually specify single core
export VLLM_CPU_OMP_THREADS_BIND=65  # Only bind to core 65
```

### Issue 2: Poor Performance Despite Correct Core Binding

**Symptoms**: Cores are bound correctly but performance is poor.

**Diagnosis**: Check for NUMA mismatch
```bash
# Check which NUMA node owns your cores
cat /sys/devices/system/cpu/cpu65/topology/physical_package_id

# Check NUMA statistics
CPID=$(podman inspect --format '{{.State.Pid}}' <container>)
numastat -p $CPID
```

**Cause**: Cores on NUMA node 0, memory on NUMA node 1 (cross-NUMA access)

**Solution**: Match cores to memory NUMA node
```bash
# Find cores on target NUMA node
cat /sys/devices/system/node/node1/cpulist

# Use cores from the same NUMA node as memory
export VLLM_CPU_OMP_THREADS_BIND=65-66  # Assuming these are on node 1
```

### Issue 3: Cores 0-1 Active When Expecting NUMA Node 1

**Symptoms**: Set `numactl --cpunodebind=1` but cores 0-1 are active.

**Cause**: `VLLM_CPU_OMP_THREADS_BIND=0-1` explicitly overrides numactl's preference.

**Solution**:
```bash
# Option A: Use auto binding
export VLLM_CPU_OMP_THREADS_BIND=auto
export CPU_VISIBLE_MEMORY_NODES=1

# Option B: Manually specify cores on node 1
NUMA1_CORES=$(cat /sys/devices/system/node/node1/cpulist)
export VLLM_CPU_OMP_THREADS_BIND=65-66  # Subset of NUMA1_CORES

# Option C: Use taskset to restrict
taskset -c $NUMA1_CORES numactl --membind=1 podman run ...
```

### Issue 4: Container Can't Start - "Not enough NUMA nodes"

**Symptoms**:
```
AssertionError: Not enough allowed NUMA nodes to bind threads of 2 local CPUWorkers.
Allowed NUMA nodes are [0].
```

**Cause**: Fewer NUMA nodes available than workers requested.

**Solution**:
```bash
# Check available NUMA nodes
numactl --hardware

# Option A: Reduce worker count
# Use fewer workers than NUMA nodes

# Option B: Make more NUMA nodes visible
export CPU_VISIBLE_MEMORY_NODES=0,1

# Option C: Use manual binding instead of auto
export VLLM_CPU_OMP_THREADS_BIND=0-31|32-63  # Manual per-worker binding
```

### Issue 5: Memory Allocation Failures

**Symptoms**: OOM errors or "cannot allocate memory" despite available RAM.

**Cause**: Memory bound to specific NUMA node that's full.

**Diagnosis**:
```bash
numastat -m  # Show per-node memory usage
```

**Solution**:
```bash
# Option A: Increase NUMA node memory or use different node
export CPU_VISIBLE_MEMORY_NODES=<node_with_more_memory>

# Option B: Remove strict memory binding (less optimal)
# Use --preferred instead of --membind
numactl --cpunodebind=1 --preferred=1 podman run ...

# Option C: Reduce KV cache size
export VLLM_CPU_KVCACHE_SPACE=20  # Reduce from default
```

---

## Performance Best Practices

### 1. Always Match Cores to NUMA Nodes

```bash
# ✅ GOOD: Cores and memory on same NUMA node
NUMA_NODE=1
CORES=$(cat /sys/devices/system/node/node${NUMA_NODE}/cpulist | cut -d- -f1-2)
export VLLM_CPU_OMP_THREADS_BIND=${CORES}
numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} vllm serve <model>

# ❌ BAD: Cross-NUMA access
export VLLM_CPU_OMP_THREADS_BIND=0-1  # Node 0
numactl --membind=1 vllm serve <model>  # Memory on node 1
```

### 2. Reserve Cores for Production Serving

```bash
# Reserve 1-2 cores for frontend/asyncio to avoid CPU oversubscription
export VLLM_CPU_OMP_THREADS_BIND=auto
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
```

### 3. Use bfloat16 for CPU

```bash
# CPU has better bfloat16 support than float16
vllm serve <model> --dtype=bfloat16
```

### 4. Set Appropriate KV Cache Size

```bash
# Benchmark to find optimal size
# Too small: fewer concurrent requests
# Too large: risk of OOM
export VLLM_CPU_KVCACHE_SPACE=40  # 40 GB
```

### 5. Monitor NUMA Statistics

```bash
# Watch for cross-NUMA traffic
watch -n 1 "numastat -p \$(pgrep -f 'vllm serve')"
```

---

## Quick Reference

### Environment Variables Summary

```bash
# CPU-specific variables
export VLLM_CPU_KVCACHE_SPACE=40           # KV cache in GB
export VLLM_CPU_OMP_THREADS_BIND=65-66     # Core binding
export VLLM_CPU_NUM_OF_RESERVED_CPU=1      # Reserved cores (auto mode only)
export CPU_VISIBLE_MEMORY_NODES=1          # NUMA nodes (auto mode only)
export VLLM_CPU_SGL_KERNEL=0               # Small-batch kernels (x86)

# General variables
export OMP_NUM_THREADS=<num>               # Only with nobind mode
```

### Common Commands

```bash
# Find NUMA topology
numactl --hardware
lscpu | grep NUMA

# Find cores on NUMA node
cat /sys/devices/system/node/node1/cpulist

# Check process affinity
taskset -cp <pid>

# Monitor per-core utilization
mpstat -P ALL 1

# Check NUMA statistics
numastat -p <pid>
```

### Code Locations

```
vllm/
├── vllm/
│   ├── envs.py                           # Environment variable definitions
│   ├── platforms/
│   │   └── cpu.py                        # CPU platform implementation
│   └── v1/
│       ├── worker/
│       │   ├── cpu_worker.py             # CPU worker (binding logic)
│       │   └── cpu_model_runner.py       # CPU model runner
│       └── attention/
│           └── backends/
│               └── cpu_attn.py           # CPU attention backend
```

---

## Additional Resources

- **vLLM CPU Documentation**: `vllm/docs/getting_started/installation/cpu.md`
- **Environment Variables**: `vllm/vllm/envs.py` (lines 696-712)
- **CPU Platform**: `vllm/vllm/platforms/cpu.py`
- **Worker Implementation**: `vllm/vllm/v1/worker/cpu_worker.py`

---

## Notes

1. **Manual vs Auto Binding**: Remember that `VLLM_CPU_NUM_OF_RESERVED_CPU` and `CPU_VISIBLE_MEMORY_NODES` **ONLY** work with `VLLM_CPU_OMP_THREADS_BIND=auto`.

2. **Binding Precedence**: `sched_setaffinity()` (used by vLLM) is stronger than `numactl --cpunodebind`. Always ensure your explicit bindings match your NUMA node.

3. **Container Layers**: When using containers with taskset/numactl, understand the layering:
   ```
   taskset → numactl → container → vLLM binding
   ```

4. **NUMA Locality**: Always verify cores and memory are on the same NUMA node for optimal performance.

5. **Reserved Cores**: Reserved cores are removed from the **end** of the sorted core list in auto mode.
