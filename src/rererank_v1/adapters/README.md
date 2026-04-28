# Adapters

本目录用于放置 Route A 的框架适配层。

目标是把成熟框架（如 LlamaIndex）与当前实验壳解耦，统一成仓库内可复用接口，例如：

- `RetrieverAdapter`
- `GeneratorAdapter`
- `VerifierAdapter`
- `ResultFormatter`

当前已经提供：

- `contracts.py`
  定义 `RetrievedEvidence`、`GenerationResult`、`VerificationResult` 以及三类 adapter 协议。

这样 `experiments/run_all.py` 等入口可以继续复用，而不必感知具体底层框架实现。
