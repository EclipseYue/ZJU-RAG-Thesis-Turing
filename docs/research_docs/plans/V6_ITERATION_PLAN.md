# V6 迭代计划：真实模型接入与安全增强

> 历史说明：本文档形成于旧路线阶段，保留作研究档案。当前正式路线请以 `docs/ROUTE_A_ARCHITECTURE.md` 为准。

## 概述

在V1–V5已建立的渐进式评测框架基础上，V6的核心目标是从“Mock评测/近似打分”过渡到“真实模型推理”，并进一步增强无答案安全机制与系统可扩展能力，使实验结果更贴近可部署形态，同时保持可复现、可对比的迭代记录方式。

## 6.1 真实模型接入

### 目标
- 用真实的BGE-reranker-base推理替代Mock重排序打分
- 对齐并验证V1–V5阶段性结论在真实模型下是否成立
- 建立可重复运行的推理与评测流水线（含失败回退策略）

### 技术实现（示意）
```python
class RealReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def score(self, query, documents):
        pairs = [[query, doc] for doc in documents]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze().cpu().numpy()
        return scores
```

### 验证策略
- 同一测试集并行跑两套：Mock vs 真实模型
- 比较宏观指标（MRR/NDCG/P@3）与每条query的差异
- 计算Mock与真实模型在“每条query表现”层面的相关性，定位近似偏差来源

## 6.2 安全机制增强

### 6.2.1 不确定性量化
为重排序输出引入不确定性估计，用于辅助拒答门控与扩展门控：
- Monte Carlo Dropout（估计认知不确定性）
- 置信区间（对分数的波动范围建模）
- 集成/多头策略（以多模型一致性评估稳定性）

### 6.2.2 动态阈值学习
将固定阈值（如PRF触发阈值、无答案阈值）替换为可学习/可自适应参数：
```python
class AdaptiveThreshold:
    def __init__(self, initial_threshold=0.15):
        self.threshold = initial_threshold
        self.feedback_history = []

    def update_threshold(self, user_feedback):
        self.feedback_history.append(user_feedback)
        self.threshold = self._calculate_optimal_threshold()
```

### 6.2.3 多因子安全评分
将无答案检测从“单阈值”扩展到“多因子融合”：
- 与已知无答案模式的语义接近度
- 查询复杂度评分（长度、歧义度、否定/约束信号）
- 文档质量与分布特征（相似度集中度、证据一致性）

## 6.3 可扩展性增强

### 6.3.1 批处理推理优化
- 重排序支持批量推理以降低单位query开销
- 向量化相似度计算以减少Python循环开销
- 对长文档进行高效分块与缓存，降低重复编码成本

### 6.3.2 缓存策略
- 对常见查询的重排序/融合结果进行缓存
- 设计合理的失效机制（语料更新、配置变更时）
- 需要时引入Redis作为分布式缓存

### 6.3.3 异步处理
- 将耗时操作异步化，降低前台等待
- PRF扩展可在后台预取/预算
- 支持实时模式与离线批处理模式切换

## 6.4 多语言与数据扩展

### 语言支持
- 增加查询语言识别与路由
- 引入多语言嵌入/重排序模型或跨语种检索策略

### 数据扩展
- 文档规模从48篇扩展到100+篇
- 增加更真实的噪声与难例（歧义/约束/无答案占比更贴近真实）

## 6.5 用户交互与反馈闭环（可选）

- 显式反馈：相关/不相关标注、拒答是否合理
- 隐式反馈：点击与停留时间、改写行为
- A/B实验：用于阈值与策略的持续优化

## 成功判据（建议）
- 真实模型接入后，核心指标在同一测试集上保持可比性，并能解释与Mock的差异来源
- 无答案检测准确率保持高水平，同时控制误拒答率
- 在100+文档规模下保持稳定运行与可接受的端到端延迟

## Phase 6.4: Multi-language Support

### 6.4.1 Language Detection
- Automatic query language identification
- Language-specific reranker models
- Cross-lingual retrieval capabilities

### 6.4.2 Evaluation Expansion
- Add multilingual document corpus (target: 100+ documents)
- Include non-English ambiguous terms
- Cross-lingual query-document pairs

## Phase 6.5: User Interaction Logging

### 6.5.1 Feedback Collection
- Implicit feedback: click-through rates, dwell time
- Explicit feedback: thumbs up/down, relevance ratings
- Query reformulation patterns

### 6.5.2 A/B Testing Framework
- Controlled experiments for threshold optimization
- User cohort analysis
- Performance impact measurement

## Implementation Timeline

### Week 1: Foundation
- [ ] Set up real model inference environment
- [ ] Implement basic real reranker integration
- [ ] Create performance benchmarking suite

### Week 2: Safety Enhancements
- [ ] Implement uncertainty quantification
- [ ] Add dynamic threshold learning
- [ ] Expand no-answer detection factors

### Week 3: Scalability
- [ ] Optimize batch processing
- [ ] Implement caching layer
- [ ] Add async processing capabilities

### Week 4: Evaluation & Refinement
- [ ] Run comprehensive V6 evaluation
- [ ] Compare mock vs real model results
- [ ] Generate V6 research report

## Expected Outcomes

### Performance Metrics
- **Real model correlation**: >0.85 Pearson correlation with mock scores
- **Latency improvement**: 50% reduction in average response time
- **Safety accuracy**: >95% no-answer detection with <5% false positives

### Dataset Expansion
- **Document count**: 48 → 100+ documents
- **Language coverage**: English + 3 additional languages
- **Query diversity**: 30 → 50+ test queries

### System Capabilities
- **Production-ready**: Full async processing with caching
- **User feedback**: Complete interaction logging system
- **A/B testing**: Framework for continuous optimization

## Risk Mitigation

### Technical Risks
- **Model loading failures**: Implement fallback to mock system
- **Memory constraints**: Progressive loading with batch processing
- **Latency degradation**: Caching and pre-computation strategies

### Evaluation Risks
- **Performance regression**: Maintain mock system for comparison
- **Dataset bias**: Expand to more diverse content sources
- **User feedback quality**: Implement validation mechanisms

## Success Criteria

### Technical Milestones
1. Real reranker integration with <100ms additional latency
2. No-answer detection accuracy >95%
3. Support for 100+ documents with sub-second response
4. Multi-language query handling for 3+ languages

### Research Contributions
1. Validation of mock-based findings with real models
2. Novel uncertainty quantification for reranking
3. Adaptive threshold learning methodology
4. Comprehensive multi-language evaluation framework

## Next Steps

1. **Environment Setup**: Install real model dependencies
2. **Baseline Measurement**: Establish current mock system performance
3. **Incremental Integration**: Phase-wise replacement of mock components
4. **Continuous Monitoring**: 历史上曾设想接入 dashboard，但该方向已不再作为当前仓库主线

---

*This plan builds upon the solid foundation established in V1-V5, transitioning from research prototype to production-ready system while maintaining the rigorous evaluation standards that validated our architectural decisions.*
