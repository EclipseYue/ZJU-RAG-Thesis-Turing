# V6迭代快速启动指南

> 历史说明：本文档是旧路线阶段计划，保留作档案参考。当前仓库正式路线请以 `docs/ROUTE_A_ARCHITECTURE.md` 与 `docs/EXPERIMENT_MASTER_GUIDE.md` 为准。

## 🚀 立即开始

### 1. 环境准备
```bash
# 安装真实模型依赖
pip install transformers torch sentence-transformers
pip install redis  # 用于缓存
pip install aiohttp  # 用于异步处理
```

### 2. 验证当前状态
```bash
# 运行V5最终验证
python reranker_study.py --phase v5 --validate

# 旧版可视化仪表板已退役，不再作为当前路线入口
```

### 3. V6核心组件启动

#### 3.1 真实模型集成（第一步）
```python
# 新建 real_reranker.py
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

class RealReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def score(self, query, documents):
        pairs = [[query, doc] for doc in documents]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, 
                               return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 提取池化输出用于评分
            scores = outputs.last_hidden_state.mean(dim=1)[:, 0]
            return torch.sigmoid(scores).numpy()
```

#### 3.2 快速测试
```python
# test_real_integration.py
from real_reranker import RealReranker
from rag_pipeline import RAGPipeline

# 对比测试
real_reranker = RealReranker()
mock_pipeline = RAGPipeline(use_mock=True)
real_pipeline = RAGPipeline(use_mock=False, reranker=real_reranker)

# 运行对比实验
results = compare_pipelines(mock_pipeline, real_pipeline)
print(f"Mock-Real Correlation: {results['correlation']}")
```

## 📊 关键监控指标

### 性能基准
- **当前Mock系统**: MRR=0.368, NDCG@5=0.494
- **目标真实系统**: 保持±5%范围内
- **延迟要求**: <500ms额外开销

### 安全检查
- **无答案检测**: 当前100%准确率（2/2）
- **误报率**: 目标<5%
- **用户满意度**: 目标>90%

## 🎯 本周行动清单

### Day 1-2: 基础集成
- [ ] 安装transformers和torch
- [ ] 实现RealReranker类
- [ ] 运行首次真实模型测试

### Day 3-4: 性能优化
- [ ] 添加批处理支持
- [ ] 实现基础缓存
- [ ] 优化内存使用

### Day 5-7: 验证与调优
- [ ] 完整V6实验运行
- [ ] 生成对比报告
- [ ] 更新可视化仪表板

## 🔧 调试工具

### 实时监控脚本
```python
# monitor.py
import time
from datetime import datetime

def monitor_performance():
    while True:
        metrics = collect_metrics()
        print(f"[{datetime.now()}] MRR: {metrics['mrr']}, Latency: {metrics['latency']}ms")
        time.sleep(60)
```

### 快速诊断命令
```bash
# 检查模型加载
python -c "from real_reranker import RealReranker; r = RealReranker(); print('OK')"

# 验证缓存
python -c "import redis; r = redis.Redis(); print(r.ping())"

# 性能测试
python -m pytest tests/test_performance.py -v
```

## 📈 可视化更新

### 新增图表
- 真实vs Mock模型对比
- 延迟分布直方图
- 缓存命中率趋势

### 仪表板增强
```javascript
// dashboard.js 新增
const realTimeChart = new Chart(ctx, {
    type: 'line',
    data: {
        datasets: [{
            label: 'Real Model Performance',
            data: realTimeData,
            borderColor: 'rgb(75, 192, 192)',
        }]
    }
});
```

## 🚨 常见问题解决

### 内存不足
```bash
# 减少批处理大小
export BATCH_SIZE=8  # 默认32

# 使用CPU模式
export CUDA_VISIBLE_DEVICES=""
```

### 模型下载慢
```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
```

### 缓存问题
```bash
# 清除Redis缓存
redis-cli FLUSHALL

# 重置缓存统计
python cache_manager.py --reset
```

## 📞 下一步行动

1. **立即执行**: 运行环境准备命令
2. **验证**: 检查V5结果并确认基线
3. **集成**: 从RealReranker开始逐步替换
4. **监控**: 启动实时监控脚本

---

**准备好开始了吗？** 先运行环境准备，然后按照Day 1-7的清单逐步推进！

遇到问题随时查看调试工具部分或创建新的issue。
