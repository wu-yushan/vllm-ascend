import bisect
from dataclasses import replace
from itertools import product

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

# bs step for the capture grid within each tier.
_DSD_BS_STEP = 2


def _is_dsd(self) -> bool:
    spec = self.vllm_config.speculative_config
    return spec is not None and spec.uses_dynamic_speculative_decoding()


def _create_padded_batch_descriptor(
    self,
    num_tokens: int,
    uniform_decode: bool,
    has_lora: bool,
    num_active_loras: int = 0,
    num_reqs: int | None = None,
) -> BatchDescriptor:
    """DSD 2-D dispatch. When num_reqs is given (catalog build), emit an exact
    (num_tokens, num_reqs) cell. On the dispatch path, pick the smallest FULL
    cell that covers the real (num_tokens, actual_bs) -- exact, then pad up to
    the next captured bs for this query_len, then PIECEWISE fallback."""
    max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs

    if _is_dsd(self):
        # Catalog build path: explicit (num_tokens, num_reqs) cell.
        if num_reqs is not None:
            return BatchDescriptor(
                num_tokens=num_tokens,
                num_reqs=min(num_reqs, max_num_seqs),
                uniform=True,
                has_lora=has_lora,
                num_active_loras=num_active_loras,
            )

        # Dispatch path. Only uniform batches can reuse a FULL graph; mixed
        # batches (e.g. a request mid-join with K_prev=0) fall through to the
        # original logic below and dispatch as PIECEWISE.
        actual_bs = self._dsd_num_reqs
        if (
            uniform_decode
            and actual_bs is not None
            and actual_bs > 0
            and num_tokens % actual_bs == 0
        ):
            query_len = num_tokens // actual_bs  # = 1 + K_prev
            full_keys = self.cudagraph_keys.get(CUDAGraphMode.FULL, set())

            def _mk(nt, nr, uni):
                return BatchDescriptor(
                    num_tokens=nt,
                    num_reqs=nr,
                    uniform=uni,
                    has_lora=has_lora,
                    num_active_loras=num_active_loras,
                )

            # 1. Exact cell: bs on the step-2 grid -> (nt, bs) is a catalog cell.
            exact_desc = _mk(num_tokens, actual_bs, True)
            if exact_desc in full_keys:
                hit = "exact"
                desc = exact_desc
            else:
                # 2. Pad up to the smallest captured bs' >= actual_bs whose
                #    query_len matches (dummy reqs fill bs' - actual_bs).
                desc = None
                bs_list = self._dsd_bs_by_ql.get(query_len)
                if bs_list:
                    idx = bisect.bisect_left(bs_list, actual_bs)
                    if idx < len(bs_list):
                        bs_pad = bs_list[idx]
                        cand = _mk(bs_pad * query_len, bs_pad, True)
                        if cand in full_keys:
                            desc = cand
                            hit = "pad"
                if desc is None:
                    hit = "piecewise"

            print(f"[DEBUG-2D] nt={num_tokens} bs={actual_bs} ql={query_len} "
                  f"hit={hit}")

            if desc is not None:
                return desc

            # 3. No FULL graph for this shape: PIECEWISE for this one step.
            num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]
            return _mk(num_tokens_padded, None, False)

    # ---- original (non-DSD / mixed-batch) logic ----
    uniform_decode_query_len = self.uniform_decode_query_len
    num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]
    if (
        uniform_decode
        and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL)
        and self.cudagraph_mode != CUDAGraphMode.FULL
    ):
        num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
        assert num_tokens_padded % uniform_decode_query_len == 0
    else:
        uniform_decode = False
        num_reqs = min(num_tokens_padded, max_num_seqs)
    return BatchDescriptor(
        num_tokens=num_tokens_padded,
        num_reqs=num_reqs,
        uniform=uniform_decode,
        has_lora=has_lora,
        num_active_loras=num_active_loras,
    )


def _dsd_2d_cells(self, uniform_decode_query_len: int):
    """2-D FULL catalog for DSD: per tier, own k + all strictly-lower k, at bs
    on a step-2 grid within the tier's [start, end] (clamped to max_num_seqs,
    inclusive of the end). Returns deduped (num_tokens, num_reqs) cells."""
    spec = self.vllm_config.speculative_config
    max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
    max_cg = self.compilation_config.max_cudagraph_capture_size or 0
    table = [t for t in (spec.num_speculative_tokens_per_batch_size or [])
             if t[2] > 0]
    # Distinct k values, descending. Lower-bs tiers are assumed to have higher k.
    all_ks = sorted({t[2] for t in table}, reverse=True)

    cells = []
    seen = set()

    def add(num_tokens, bs):
        if max_cg and num_tokens > max_cg:
            return
        if not (1 <= bs <= max_num_seqs):
            return
        cell = (num_tokens, bs)
        if cell not in seen:
            seen.add(cell)
            cells.append(cell)

    for start, end, k_own in table:
        # Capture own k + every strictly-lower k (reachable as K_prev when bs
        # decreases out of a higher-bs tier that uses that lower k).
        ks = [k for k in all_ks if k <= k_own]
        bs_lo = max(start, 1)
        bs_hi = min(end, max_num_seqs)
        if bs_hi < bs_lo:
            continue
        # Step-2 grid, always including bs_hi.
        bs_vals = list(range(bs_lo, bs_hi + 1, _DSD_BS_STEP))
        if not bs_vals or bs_vals[-1] != bs_hi:
            bs_vals.append(bs_hi)
        for k in ks:
            qlen = k + 1
            for bs in bs_vals:
                add(bs * qlen, bs)
    return cells


def initialize_cudagraph_keys(
    self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int = 1
):
    """DSD: build a 2-D FULL catalog {(num_tokens, num_reqs) cells} instead of
    the 1-D fixed-K FULL keys. Non-DSD: original behavior unchanged."""
    self.cudagraph_mode = cudagraph_mode
    # [DSD 2-D] per-step actual num_reqs side-channel: written by the model
    # runner's dispatch_cudagraph right before each dispatch() and read in
    # _create_padded_batch_descriptor (DSD needs actual_bs to back out
    # query_len = num_tokens // actual_bs, since K varies per step).
    self._dsd_num_reqs = None
    # [DSD 2-D] bs grid per query_len, for dispatch padding (round actual_bs up
    # to the next captured bs with the same query_len).
    self._dsd_bs_by_ql: dict[int, list[int]] = {}

    if cudagraph_mode == CUDAGraphMode.NONE:
        self.keys_initialized = True
        return

    self._compute_bs_to_padded_graph_size()
    lora_cases = self._get_lora_cases()
    self.captured_lora_counts = [c for c in lora_cases if c]
    print(f"[DEBUG] cudagraph_mode.decode_mode() = {cudagraph_mode.decode_mode()}")
    # ---- mixed-mode PIECEWISE keys (prefill fallback) — original ----
    if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
        assert self.compilation_config.cudagraph_capture_sizes is not None, (
            "Cudagraph capture sizes must be set when mixed mode is enabled."
        )
        for bs, num_active_loras in product(
            self.compilation_config.cudagraph_capture_sizes, lora_cases
        ):
            batch_desc = self._create_padded_batch_descriptor(
                bs, False, num_active_loras > 0, num_active_loras
            )
            if cudagraph_mode.mixed_mode() == CUDAGraphMode.PIECEWISE:
                batch_desc = replace(batch_desc, num_reqs=None, uniform=False)
            self.add_cudagraph_key(cudagraph_mode.mixed_mode(), batch_desc)

    # ---- FULL decode keys ----
    if (
        cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        and cudagraph_mode.separate_routine()
    ):
        if _is_dsd(self):
            spec = self.vllm_config.speculative_config
            # Safety: each K-tier's max num_tokens must fit within capture range.
            max_cg_size = self.compilation_config.max_cudagraph_capture_size
            for _, _, k in (
                spec.num_speculative_tokens_per_batch_size or []
            ):
                if k > 0:
                    tier_max = (k + 1) * self.vllm_config.scheduler_config.max_num_seqs
                    assert tier_max <= max_cg_size, (
                        f"DSD 2-D: K={k} tier max num_tokens={tier_max} exceeds "
                        f"max_cudagraph_capture_size={max_cg_size}. Increase "
                        f"max_cudagraph_capture_size or reduce max_num_seqs.")
            # 2-D catalog: one FULL graph per (num_tokens, num_reqs) cell.
            dsd_cells = _dsd_2d_cells(self, uniform_decode_query_len)
            # Precompute bs grid per query_len for dispatch padding.
            for nt, nr in dsd_cells:
                ql = nt // nr  # nr > 0 guaranteed by add()
                self._dsd_bs_by_ql.setdefault(ql, []).append(nr)
            for ql in self._dsd_bs_by_ql:
                self._dsd_bs_by_ql[ql].sort()
            print(f"[DEBUG-2D] DSD FULL catalog: {len(dsd_cells)} cells = "
                  f"{sorted(dsd_cells)}")
            print(f"[DEBUG-2D] DSD bs grid per ql: {self._dsd_bs_by_ql}")
            for num_tokens, num_reqs in dsd_cells:
                for num_active_loras in lora_cases:
                    self.add_cudagraph_key(
                        CUDAGraphMode.FULL,
                        self._create_padded_batch_descriptor(
                            num_tokens,
                            True,
                            num_active_loras > 0,
                            num_active_loras,
                            num_reqs=num_reqs,
                        ),
                    )
        else:
            # original 1-D FULL keys (fixed K)
            max_num_tokens = (
                uniform_decode_query_len
                * self.vllm_config.scheduler_config.max_num_seqs
            )
            assert self.compilation_config.cudagraph_capture_sizes is not None, (
                "Cudagraph capture sizes must be set when full mode is enabled."
            )
            cudagraph_capture_sizes_for_decode = [
                x
                for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]
            for bs, num_active_loras in product(
                cudagraph_capture_sizes_for_decode, lora_cases
            ):
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(
                        bs, True, num_active_loras > 0, num_active_loras
                    ),
                )

    self.keys_initialized = True


CudagraphDispatcher._create_padded_batch_descriptor = _create_padded_batch_descriptor
CudagraphDispatcher.initialize_cudagraph_keys = initialize_cudagraph_keys
