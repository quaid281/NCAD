---
name: Code & Test Hygiene, Modularization & Anti-Bloat Guard
description: "CRITICAL TRIGGER: MUST read and execute whenever creating, editing, or refactoring Python modules in src/, scripts/, or tests/. MANDATORY OUTPUT: Verify <= 800 LOC per file ([INV-800LOC]), <= 20 files per directory, memory/autograd hygiene, and report passing pytest verification in chat."
runtime_sop:
  - "[INV-800LOC] Enforce <= 800 LOC per file (Facade pattern) and <= 20 files per directory across src/ and tests/ with atomic state writes."
  - "[INV-ELEGANCE] All solutions must be elegant: resolve defects at the root cause; zero band-aids/shims; favor minimal moving parts and vectorized idioms."
  - "Detach autograd graphs outside backward passes (.detach()), snapshot checkpoints to host CPU (.detach().cpu().clone()), and guard PyTorch hooks/executors with try/finally."
  - "Ban df.iterrows() and silent except Exception: pass; require 100% passing pytest verification with real mathematical assertions."
---

# Code & Test Hygiene, Modularization & Anti-Bloat Guard Skill

This skill governs structural codebase modularity, runtime memory/resource hygiene, CUDA allocator & asyncio backpressure stability, Pandas Copy-on-Write block management, PyTorch hook/logging handler lifecycle, GPU/vectorized kernel latency, and test-suite integrity across `src/`, `scripts/`, and `tests/`.

> **Progressive Disclosure Reference**:
> - [`references/memory_cuda_and_kernel_hygiene.md`](file:///C:/Users/andre/OneDrive/Desktop/NCAD_CS/.agents/skills/code_hygiene_and_modularization/references/memory_cuda_and_kernel_hygiene.md): Deep Memory, CUDA Anti-Fragmentation, Asyncio Backpressure & Vectorized Kernels (SOP 5 & SOP 6).

---

## 1. Standard Operating Procedures (SOPs)

### SOP 1: The 800 LOC Module & Test Decomposition Protocol (`[INV-800LOC]`)
1. **Production Modules (`src/`, `scripts/`) — The Facade Package Pattern**: When a module approaches or exceeds 800 LOC, convert `module_name.py` into a package directory `module_name/` with submodules ($< 400$ LOC each) and re-export all public symbols in `__init__.py` via `__all__`.
2. **Test Files (`tests/`) — Behavioral Slice Decomposition**: Split large test files into a subdirectory with shared fixtures in `conftest.py` ($< 150$ LOC) and independent behavioral slices ($\le 500$ LOC each).

### SOP 2: The 20-File Directory Density Ceiling
No folder in `src/` or `tests/` may contain more than **20 files** (unlimited subdirectories permitted). Group files by domain or architectural layer (`models/`, `data/`, `training/`, `evaluation/`, `utils/`).

### SOP 3: Fail-Closed & Atomic State Persistence
1. **Atomic File Writes**: Never write critical state, benchmark CSVs, or checkpoints with raw `open(path, "w")` without atomic replacement when updating shared artifacts. Write to a PID-scoped temporary file in the same directory, flush (`os.fsync()`), and atomically swap via `os.replace()`.
2. **Fail-Closed Error Policy & `df.iterrows()` Ban**:
   - `except Exception: pass` is **strictly banned**; fail closed and log structured diagnostics.
   - Never use `df.iterrows()` (triggers Pandas 2.x/3.x Cython `ObjectEngine` segfaults); iterate over `df.to_dict(orient="records")` or execute `[B, T, D]` tensor batching.

### SOP 4: Test Suite Anti-Bloat, Fast Scaling & Real Assertion Discipline
1. **Dead Test & Legacy Import Purging (`[INV-MIGRATION-PURGE]`)**: Import from canonical production paths only; immediately delete orphaned test files when deprecated modules are purged.
2. **Fast Scaling**: Use `hidden_dim=8`, `num_layers=1` and compact synthetic streams ($T = 32\text{--}128$) with deterministic seeds (`42`); mark tests $> 2\text{ s}$ with `@pytest.mark.integration`.
3. **Real Mathematical Assertions & Singleton Isolation**: Ban `assert True`; verify Crouzeix contractivity bounds, Stiefel orthogonality ($\|\mathbf{W}^\top\mathbf{W} - \mathbf{I}\|_F < \epsilon$), positive-definite stress cones ($\lambda_{\min}(\boldsymbol{\Sigma}) \ge \gamma$), Littlewood-Paley partition of unity, and degree-1 scale homogeneity ($D(c \cdot e) = c \cdot D(e)$). Reset any patched module-level singletons via `autouse=True` fixtures.

### SOP 5: Memory Leak Prevention, CUDA Anti-Fragmentation & Asyncio Backpressure
*(Full implementation guide: [`references/memory_cuda_and_kernel_hygiene.md`](file:///C:/Users/andre/OneDrive/Desktop/NCAD_CS/.agents/skills/code_hygiene_and_modularization/references/memory_cuda_and_kernel_hygiene.md))*
1. **Autograd & Checkpoints**: Detach stored tensors (`z_tgt.detach()`); wrap evaluation and SPOT calibration in `torch.inference_mode()`; snapshot best `state_dict` to host CPU via `{k: v.detach().cpu().clone() ...}`.
2. **CUDA Bucketing**: Pad dynamic batch evaluations to power-of-two sizes ($2^{\lceil \log_2 B \rceil}$); never call `torch.cuda.empty_cache()` inside per-stream benchmark loops.
3. **Asyncio & Workers**: Store fire-and-forget tasks in a strong-reference set (`self._bg_tasks.add(t); t.add_done_callback(self._bg_tasks.discard)`); bound `asyncio.Queue(maxsize=...)`.
4. **Pandas 3.0 CoW & PyTorch `Dataset` Buffers**: Call `.copy()` after pruning long-lived lookup caches; extract contiguous `np.float32` arrays via `.to_numpy(dtype=np.float32, copy=True)` in `Dataset.__init__` and release `self.df = None`.
5. **Hooks, Handlers & Executors**: Guard forward hooks with `try ... finally: hook.remove()`, close `logging.FileHandler` / `sqlite3` / file handles deterministically, and tear down thread pools via `executor.shutdown(wait=False, cancel_futures=True)`.

### SOP 6: GPU Acceleration, Tensor Batching & Vectorized Kernel Mandate
- Execute rFFT dyadic-shell filtering, Hankel resolvent traces (`torch.linalg.eigvalsh`), and Reynolds stress covariance closures via batched `[B, T, D]` tensor kernels on `cuda` when available; never loop over time steps or channels in Python when vectorized tensor/Toeplitz operators apply.

### SOP 7: The Principle of Elegance & Minimal Sufficient Complexity (`[INV-ELEGANCE]`)
1. **Root-Cause Purity**: Resolve defects strictly at their origin (pipeline, schema, or architectural layer); zero downstream `if x is None` band-aids or stream-specific hacks.
2. **Minimal Moving Parts & Vectorized Clarity**: Reject unearned wrapper classes, duplicate forward passes, and speculative indirection; express quantitative logic via clean, self-documenting matrix/tensor idioms.

---

## 2. Audit & Verification CLI Reference
Execute from `workdir="C:/Users/andre/OneDrive/Desktop/NCAD_CS"`:
```powershell
& "C:\Users\andre\anaconda3\envs\alpha\python.exe" -m pytest tests/ -q
& "C:\Users\andre\anaconda3\envs\alpha\python.exe" scripts/build_and_verify_manuscript.py
```
