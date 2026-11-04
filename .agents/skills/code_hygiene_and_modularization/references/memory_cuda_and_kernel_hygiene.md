# Memory Leak Prevention, CUDA Anti-Fragmentation, Asyncio Backpressure & Vectorized Kernels

This reference manual provides the complete implementation rules for **SOP 5** and **SOP 6** of [`code_hygiene_and_modularization/SKILL.md`](file:///C:/Users/andre/OneDrive/Desktop/NCAD_CS/.agents/skills/code_hygiene_and_modularization/SKILL.md).

---

## 1. SOP 5: Memory Leak Prevention, CUDA Anti-Fragmentation & Resource Hygiene

1. **PyTorch Autograd Graph Detachment & Host-CPU Checkpoint Snapshots**:
   - Never store tensors carrying `requires_grad=True` into class attributes (`self._last_z = z.detach()`), telemetry buffers, or history lists.
   - Wrap all evaluation, anomaly scoring, and SPOT calibration forward passes in `with torch.inference_mode():` or `with torch.no_grad():`.
   - In training loops, always offload best validation `state_dict` snapshots to host RAM via `{k: v.detach().cpu().clone() for k, v in model.state_dict().items()}`—never call `copy.deepcopy(model.state_dict())` or `v.clone()` directly in GPU VRAM.
2. **CUDA Caching Allocator Anti-Fragmentation**:
   - In variable-length stream evaluations, pad batch inputs to power-of-two batch buckets ($B_{\text{bucket}} = 2^{\lceil \log_2 B_{\text{active}} \rceil}$) so PyTorch's CUDA caching allocator reuses $\le \lceil \log_2 B \rceil$ block sizes instead of splitting VRAM across $O(B)$ shrinking shapes.
   - Never call `torch.cuda.empty_cache()` inside per-stream benchmark loops (which forces `cudaDeviceSynchronize` and `cudaMalloc` thrashing); flush once post-benchmark if needed.
3. **Asyncio Strong-Reference Task Sets & Bounded Queues**:
   - Because Python's `asyncio` event loop holds only weak references to tasks, every fire-and-forget `asyncio.create_task()` / `loop.create_task()` **must** save the task in a strong-reference set with auto-discard (`self._bg_tasks.add(t); t.add_done_callback(self._bg_tasks.discard)`).
   - Every `asyncio.Queue` must enforce a finite `maxsize` with drop-oldest eviction on `QueueFull`.
4. **Bounded LRU/TTL Caches & Deterministic Handle Closure**:
   - Enforce FIFO/LRU capacity ceilings (`_MAX_CACHE_ENTRIES = 16`–`500`) on module-level data/preprocessor caches.
   - Always close Matplotlib figures (`plt.close(fig)`) after saving to prevent memory leaks in multi-stream plotting scripts.
5. **Pandas 3.0 Copy-on-Write (`BlockManager`) Pinning & PyTorch `Dataset` Materialization**:
   - Under Pandas $\ge 3.0$ Copy-on-Write (`mode.copy_on_write = True`), row slices (`df.iloc[...]`), boolean masks (`df[mask]`), column subsets (`df[cols]`), and `df.groupby(...)` objects hold `BlockValuesRefs` links that keep the entire parent `BlockManager` pinned in RAM. Always call `.copy()` after column-pruning or row-filtering long-lived lookup caches.
   - PyTorch `Dataset` classes must extract contiguous `np.float32` arrays via `.to_numpy(dtype=np.float32, copy=True)` in `__init__`, release the source DataFrame (`self.df = None`), and never execute `df.loc[...]` inside `__getitem__`.
6. **PyTorch Forward-Hook & Logging FileHandler Lifecycle**:
   - Never call `module.register_forward_hook(...)` without an enclosing `try ... finally: hook.remove()` context manager, and always call `.remove()` on prior handles before clearing `self.hooks`.
   - Never instantiate `logging.FileHandler` / `RotatingFileHandler` before checking `if not logger.handlers:` (since `FileHandler.__init__` immediately opens an OS file descriptor), and always call `h.close()` when removing handlers from a logger.
7. **`ThreadPoolExecutor`, `ProcessPoolExecutor` & `subprocess` Anti-Zombie Lifecycle**:
   - Never pair `as_completed(..., timeout=...)` or `future.result(timeout=...)` with a blocking `with ThreadPoolExecutor(...) as executor:` context manager (whose `__exit__` calls `shutdown(wait=True)` and deadlocks on hung workers). Always tear down via `try: ... finally: executor.shutdown(wait=False, cancel_futures=True)`.
   - Every `subprocess.run(...)` and `subprocess.check_output(...)` call must enforce an explicit finite `timeout=...`.

---

## 2. SOP 6: GPU Acceleration, Tensor Batching & Vectorized Kernel Mandate

1. **Zero Per-Step Host-Device Sync Stalls**:
   - Never call `.cpu().numpy()`, `.cpu().item()`, `np.random.choice`, or `np.random.normal` inside inner window/stream loops.
   - Compute window scores directly on the target device via fused PyTorch kernels, transferring flat 1D score arrays to host memory once per stream.
2. **Vectorized Matrix, Spectral & Toeplitz Operators Over Nested Loops**:
   - Replace nested Python loops over time lags, frequency shells, or channel pairs with vectorized 1D slice indexing, rFFT broadcasting, `torch.linalg.eigvalsh`, and Toeplitz matrix multiplications.
3. **True `[B, T, D]` Batch Tensor Inference & Dynamic CUDA Binding**:
   - Batch evaluation methods must process full `[B, T, D]` window tensors in a single forward pass without duplicate submodule invocations.
   - Default neural engine device binding to `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
