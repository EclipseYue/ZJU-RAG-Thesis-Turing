"""
Real Reranker Implementation for V6 Experiment
基于真实BGE-reranker-base模型的重排序实现
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealReranker:
    """
    真实重排序器实现
    使用BAAI/bge-reranker-base模型进行文档重排序
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = None):
        """
        初始化真实重排序器
        
        Args:
            model_name: 模型名称
            device: 计算设备 ('cuda', 'cpu', 或自动检测)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading real reranker model: {model_name} on {self.device}")
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # 模型参数
        self.max_length = 512
        self.batch_size = 8
        
        logger.info("Real reranker initialized successfully")
    
    def score(self, query: str, documents: List[str]) -> np.ndarray:
        """
        对文档进行重排序评分
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            
        Returns:
            文档相关性评分数组
        """
        if not documents:
            return np.array([])
        
        start_time = time.time()
        
        # 构建查询-文档对
        pairs = [[query, doc] for doc in documents]
        
        # 批量处理以提高效率
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            batch_scores = self._score_batch(batch_pairs)
            scores.extend(batch_scores)
        
        scores = np.array(scores)
        
        elapsed = time.time() - start_time
        logger.info(f"Scored {len(documents)} documents in {elapsed:.3f}s")
        
        return scores
    
    def _score_batch(self, pairs: List[List[str]]) -> List[float]:
        """
        批量评分
        
        Args:
            pairs: 查询-文档对列表
            
        Returns:
            评分列表
        """
        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 获取logits并转换为概率
            logits = outputs.logits
            # BGE reranker输出的是相关性分数，取正类概率
            scores = torch.sigmoid(logits[:, 1] if logits.shape[1] > 1 else logits.squeeze())
        
        return scores.cpu().numpy().tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "parameters": sum(p.numel() for p in self.model.parameters())
        }


class MockReranker:
    """
    Mock重排序器，用于与真实模型对比
    模拟BGE-reranker的评分行为
    """
    
    def __init__(self):
        self.name = "MockReranker"
    
    def score(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Mock评分函数，模拟真实模型的行为
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            
        Returns:
            Mock评分数组
        """
        import random
        
        # 模拟真实模型的评分分布
        base_score = 0.6
        scores = []
        
        for doc in documents:
            # 基于文档长度和关键词匹配模拟评分
            length_factor = min(len(doc) / 1000, 1.0)
            keyword_match = sum(1 for word in query.split() 
                              if word.lower() in doc.lower()) / max(len(query.split()), 1)
            
            # 添加随机噪声模拟真实场景
            noise = random.gauss(0, 0.1)
            score = base_score + 0.3 * keyword_match + 0.1 * length_factor + noise
            score = max(0, min(1, score))  # 限制在[0,1]范围
            
            scores.append(score)
        
        return np.array(scores)


# 兼容性接口
class RerankerFactory:
    """重排序器工厂类"""
    
    @staticmethod
    def create_reranker(reranker_type: str = "real", **kwargs) -> Any:
        """
        创建重排序器实例
        
        Args:
            reranker_type: "real" 或 "mock"
            **kwargs: 传递给具体类的参数
            
        Returns:
            重排序器实例
        """
        if reranker_type == "real":
            return RealReranker(**kwargs)
        elif reranker_type == "mock":
            return MockReranker()
        else:
            raise ValueError(f"Unsupported reranker type: {reranker_type}")


if __name__ == "__main__":
    # 测试代码
    reranker = RealReranker()
    
    # 测试数据
    query = "What is machine learning?"
    docs = [
        "Machine learning is a subset of artificial intelligence...",
        "Deep learning uses neural networks with multiple layers...",
        "Python is a popular programming language for data science..."
    ]
    
    scores = reranker.score(query, docs)
    print("Scores:", scores)
    print("Model info:", reranker.get_model_info())