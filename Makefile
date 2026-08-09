# mma_half.cu -- FP16 GEMM for H100 (sm_90a)
#
#   make            build everything
#   make check      correctness: 57 shapes + fp64-anchored + large/odd-N
#   make perf       performance: per-shape vs cuBLAS, and the full table
#   make san        compute-sanitizer: memcheck/racecheck/synccheck/initcheck
#   make probe      re-derive the wgmma descriptor encodings from scratch
#   make clean
#
# NOTE the `a` in sm_90a -- plain sm_90 rejects wgmma.
# NOTE benchmarks must run SERIALLY. Two of these on one GPU contend and produce
#      meaningless numbers (cuBLAS reading 265 TF at 8192^3 instead of ~790).

NVCC    ?= nvcc
# Path to a CUTLASS 3.x/4.x checkout, used only by the `cutlass` / `three_way` targets.
# Override on the command line:  make cutlass CUTLASS=/path/to/cutlass
CUTLASS ?= $(HOME)/cutlass
ARCH    := -gencode arch=compute_90a,code=sm_90a
CXXFLAGS:= $(ARCH) -O3 -I.
LDLIBS  := -lcublas
SANITIZER ?= compute-sanitizer

CHECK   := check_all check_fp64 check_large
PERF    := perf_all perf_table perf_cmp perf_prof smem_budget
SAN     := san_sweep san_large
PROBE   := probe_kmajor probe_mnmajor probe_mnmajor_kadv
# -DNDEBUG is REQUIRED, not cosmetic: without it CUTLASS's device asserts block inlining and
# ptxas serializes the wgmma pipeline (warning C7510), costing ~10-100% depending on shape.
CUTFLAGS:= -std=c++17 -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include \
           --expt-relaxed-constexpr -DNDEBUG

ALL := $(CHECK) $(PERF) $(SAN) $(PROBE)

.PHONY: all check perf san probe clean
all: $(ALL)

# probes are standalone -- they do not include the kernel, and need the driver API
probe_kmajor probe_mnmajor probe_mnmajor_kadv: %: %.cu
	$(NVCC) $(CXXFLAGS) -o $@ $< -lcuda

# sanitizer + profiling targets do not link cuBLAS
san_sweep san_large perf_prof smem_budget: %: %.cu mma_half.cu
	$(NVCC) $(CXXFLAGS) -o $@ $<

%: %.cu mma_half.cu
	$(NVCC) $(CXXFLAGS) -o $@ $< $(LDLIBS)

check: $(CHECK)
	@echo "=== 57 shapes vs cuBLAS ==="        && ./check_all
	@echo "=== 15 shapes vs fp64 host ref ===" && ./check_fp64 10
	@echo "=== large + odd-N vs cuBLAS ==="    && ./check_large

perf: $(PERF)
	@echo "=== per-shape vs cuBLAS ==="        && ./perf_all
	@echo "=== headline table ==="             && ./perf_table

san: $(SAN)
	@for t in memcheck racecheck synccheck initcheck; do \
	  printf "%-11s: " $$t; $(SANITIZER) --tool $$t ./san_sweep 2>&1 | grep -E "ERROR SUMMARY|RACECHECK SUMMARY" | tail -1; \
	done
	@printf "%-11s: " "memcheck/lg"; $(SANITIZER) --tool memcheck ./san_large 2>&1 | grep "ERROR SUMMARY"

probe: $(PROBE)
	@echo "=== K-major descriptor (A, and B before step 5) ===" && ./probe_kmajor  | head -2
	@echo "=== MN-major descriptor (B, native N-major) ==="     && ./probe_mnmajor | head -2
	@echo "=== MN-major k-advance ==="                          && ./probe_mnmajor_kadv

# CUTLASS reference build. Fails with a readable message rather than a wall of missing-header
# errors when CUTLASS points somewhere that does not exist.
three_way: three_way.cu mma_half.cu mma_half_cutlass.cu perf_shapes.h
	@test -f "$(CUTLASS)/include/cutlass/cutlass.h" || { \
	  echo "error: CUTLASS not found at '$(CUTLASS)'."; \
	  echo "       Set it, e.g.  make cutlass CUTLASS=/path/to/cutlass"; exit 1; }
	$(NVCC) $(CXXFLAGS) $(CUTFLAGS) -DCUTLASS_NO_SOLVE -o $@ three_way.cu mma_half_cutlass.cu $(LDLIBS)

cutlass: three_way
	./three_way

clean:
	rm -f $(ALL) three_way
