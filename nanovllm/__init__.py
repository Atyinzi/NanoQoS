from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    # Keep CPU-only scheduler tests independent of CUDA/FlashAttention imports.
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(name)
