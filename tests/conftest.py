"""Session-wide GPU memory hygiene: this suite loads dozens of real models
(often several GB each, more with CUDA graph capture's private pools) across
many test files sharing one long-lived pytest process. PyTorch's caching
allocator doesn't return freed blocks to the driver on its own, and several
individual fixtures don't explicitly release their model between tests --
across the full suite that accumulates enough to exhaust a 12GB card even
though no single test actually needs that much at once. Returning unused
cached blocks after every test keeps peak usage bounded by whatever the
single heaviest test needs, not the sum of everything that ever ran.
"""
import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def _release_cuda_memory_after_each_test():
    yield
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
