"""
V6实验：真实模型 vs Mock模型对比实验
本科毕业论文实验验证部分
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入实验模块
from rererank_v1.rag_pipeline import RAGPipeline
from rererank_v1.benchmark_data import DOCUMENTS, TEST_CASES
from rererank_v1.metrics import calculate_mrr, calculate_ndcg, calculate_precision
from rererank_v1.paths import data_dir, results_dir


class V6Experiment:
    """
    V6实验：真实模型与Mock模型的全面对比分析
    """
    
    def __init__(self):
        self.experiment_name = "V6_Real_vs_Mock_Comparison"
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.results = {}
        
    def setup_experiment(self):
        """设置实验环境"""
        logger.info("=" * 60)
        logger.info("V6实验：真实模型 vs Mock模型对比")
        logger.info("=" * 60)
        
        # 获取测试数据
        self.documents = DOCUMENTS
        self.test_cases = TEST_CASES
        
        logger.info(f"文档数量: {len(self.documents)}")
        logger.info(f"查询数量: {len(self.test_cases)}")

    def _get_relevant_ranks(self, results_list: List[Dict[str, Any]], relevant_doc_ids: List[int]) -> List[int]:
        ranks = []
        for doc_id in relevant_doc_ids:
            found_rank = next((i + 1 for i, item in enumerate(results_list) if item.get('id') == doc_id), 0)
            if found_rank > 0:
                ranks.append(found_rank)
        return sorted(ranks)

    def _evaluate_one_pipeline(self, rag: RAGPipeline) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        per_case = []

        mrr_retrieval, mrr_rerank, mrr_final = [], [], []
        ndcg_retrieval, ndcg_rerank, ndcg_final = [], [], []
        p3_retrieval, p3_rerank, p3_final = [], [], []

        for case in self.test_cases:
            t0 = time.time()

            retrieved_docs = rag._retrieve(case.query, top_k=20)
            ranks_retrieval = self._get_relevant_ranks(retrieved_docs, case.relevant_doc_ids)

            reranked_docs = rag._rerank(case.query, retrieved_docs)
            ranks_rerank = self._get_relevant_ranks(reranked_docs, case.relevant_doc_ids)

            final_results = rag.search(case.query, top_k=10)
            ranks_final = self._get_relevant_ranks(final_results, case.relevant_doc_ids)

            mrr_retrieval.append(calculate_mrr(ranks_retrieval))
            mrr_rerank.append(calculate_mrr(ranks_rerank))
            mrr_final.append(calculate_mrr(ranks_final))

            ndcg_retrieval.append(calculate_ndcg(ranks_retrieval, k=5))
            ndcg_rerank.append(calculate_ndcg(ranks_rerank, k=5))
            ndcg_final.append(calculate_ndcg(ranks_final, k=5))

            p3_retrieval.append(calculate_precision(ranks_retrieval, k=3))
            p3_rerank.append(calculate_precision(ranks_rerank, k=3))
            p3_final.append(calculate_precision(ranks_final, k=3))

            per_case.append({
                "query": case.query,
                "description": case.description,
                "relevant_doc_ids": case.relevant_doc_ids,
                "ranks": {
                    "retrieval": ranks_retrieval,
                    "rerank": ranks_rerank,
                    "final": ranks_final
                },
                "metrics": {
                    "mrr_retrieval": mrr_retrieval[-1],
                    "mrr_rerank": mrr_rerank[-1],
                    "mrr_final": mrr_final[-1],
                    "ndcg_retrieval": ndcg_retrieval[-1],
                    "ndcg_rerank": ndcg_rerank[-1],
                    "ndcg_final": ndcg_final[-1],
                    "p3_retrieval": p3_retrieval[-1],
                    "p3_rerank": p3_rerank[-1],
                    "p3_final": p3_final[-1],
                },
                "latency_s": time.time() - t0
            })

        summary = {
            "mrr_retrieval": float(np.mean(mrr_retrieval)) if mrr_retrieval else 0.0,
            "mrr_rerank": float(np.mean(mrr_rerank)) if mrr_rerank else 0.0,
            "mrr_final": float(np.mean(mrr_final)) if mrr_final else 0.0,
            "ndcg_retrieval": float(np.mean(ndcg_retrieval)) if ndcg_retrieval else 0.0,
            "ndcg_rerank": float(np.mean(ndcg_rerank)) if ndcg_rerank else 0.0,
            "ndcg_final": float(np.mean(ndcg_final)) if ndcg_final else 0.0,
            "precision_retrieval": float(np.mean(p3_retrieval)) if p3_retrieval else 0.0,
            "precision_rerank": float(np.mean(p3_rerank)) if p3_rerank else 0.0,
            "precision_final": float(np.mean(p3_final)) if p3_final else 0.0,
        }

        return summary, per_case
        
    def run_mock_experiment(self) -> Dict[str, Any]:
        """运行Mock模型实验"""
        logger.info("\n🔍 运行Mock模型实验...")
        
        os.environ['FORCE_MOCK'] = '1'

        # 创建Mock管道
        mock_pipeline = RAGPipeline(use_v6_reranker=False)
        mock_pipeline.add_documents(self.documents)
        
        # 记录开始时间
        start_time = time.time()
        
        mock_metrics, mock_results = self._evaluate_one_pipeline(mock_pipeline)
        
        # 记录结束时间
        elapsed_time = time.time() - start_time
        
        return {
            "model_type": "mock",
            "pipeline_mode": {
                "force_mock": True,
                "mock_mode": bool(getattr(mock_pipeline, "mock_mode", False)),
                "use_v6_reranker": bool(getattr(mock_pipeline, "use_v6_reranker", False)),
                "device": getattr(mock_pipeline, "device", None)
            },
            "metrics": mock_metrics,
            "results": mock_results,
            "total_time": elapsed_time,
            "avg_time_per_query": elapsed_time / max(len(self.test_cases), 1)
        }
    
    def run_real_experiment(self) -> Dict[str, Any]:
        """运行真实模型实验"""
        logger.info("\n🤖 运行真实模型实验...")
        
        os.environ['FORCE_MOCK'] = '0'
        if os.getenv("V6_ALLOW_DOWNLOAD", "0") != "1":
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # 创建真实管道
        real_pipeline = RAGPipeline(use_v6_reranker=True)
        real_pipeline.add_documents(self.documents)
        
        # 记录开始时间
        start_time = time.time()
        
        real_metrics, real_results = self._evaluate_one_pipeline(real_pipeline)
        
        # 记录结束时间
        elapsed_time = time.time() - start_time
        
        return {
            "model_type": "real",
            "pipeline_mode": {
                "force_mock": False,
                "mock_mode": bool(getattr(real_pipeline, "mock_mode", False)),
                "use_v6_reranker": bool(getattr(real_pipeline, "use_v6_reranker", False)),
                "device": getattr(real_pipeline, "device", None)
            },
            "metrics": real_metrics,
            "results": real_results,
            "total_time": elapsed_time,
            "avg_time_per_query": elapsed_time / max(len(self.test_cases), 1)
        }
    
    def analyze_correlation(self, mock_scores: List[float], real_scores: List[float]) -> Dict[str, float]:
        """分析Mock与真实模型的相关性（不依赖SciPy）"""
        if len(mock_scores) != len(real_scores) or len(mock_scores) == 0:
            return {"pearson": 0.0, "spearman": 0.0}

        x = np.asarray(mock_scores, dtype=float)
        y = np.asarray(real_scores, dtype=float)

        x = x - x.mean()
        y = y - y.mean()
        denom = (np.linalg.norm(x) * np.linalg.norm(y))
        pearson = float((x @ y) / denom) if denom > 0 else 0.0

        def rankdata(a: np.ndarray) -> np.ndarray:
            order = a.argsort()
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(a) + 1, dtype=float)
            return ranks

        rx = rankdata(np.asarray(mock_scores, dtype=float))
        ry = rankdata(np.asarray(real_scores, dtype=float))
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        denom_r = (np.linalg.norm(rx) * np.linalg.norm(ry))
        spearman = float((rx @ ry) / denom_r) if denom_r > 0 else 0.0

        if np.isnan(pearson):
            pearson = 0.0
        if np.isnan(spearman):
            spearman = 0.0

        return {"pearson": pearson, "spearman": spearman}
    
    def generate_comparison_report(self, mock_data: Dict, real_data: Dict) -> Dict[str, Any]:
        """生成对比分析报告"""
        
        # 使用“每条query的最终MRR”做相关性分析（不依赖对齐的doc-level打分）
        mock_scores = [r["metrics"]["mrr_final"] for r in mock_data.get("results", []) if r.get("metrics")]
        real_scores = [r["metrics"]["mrr_final"] for r in real_data.get("results", []) if r.get("metrics")]
        
        # 计算相关性
        correlation = self.analyze_correlation(mock_scores, real_scores)
        
        # 性能对比
        mock_avg = float(mock_data.get("avg_time_per_query", 0.0))
        real_avg = float(real_data.get("avg_time_per_query", 0.0))
        performance_comparison = {
            "mrr_improvement": (
                (real_data["metrics"]["mrr_final"] - mock_data["metrics"]["mrr_final"])
                / mock_data["metrics"]["mrr_final"] * 100
            ) if mock_data["metrics"]["mrr_final"] > 0 else 0,
            "ndcg_improvement": (
                (real_data["metrics"]["ndcg_final"] - mock_data["metrics"]["ndcg_final"])
                / mock_data["metrics"]["ndcg_final"] * 100
            ) if mock_data["metrics"]["ndcg_final"] > 0 else 0,
            "time_overhead_pct": (
                (real_avg - mock_avg) / mock_avg * 100
            ) if mock_avg > 1e-6 else 0.0,
            "time_overhead_s": real_avg - mock_avg
        }
        
        return {
            "correlation_analysis": correlation,
            "performance_comparison": performance_comparison,
            "mock_vs_real": {
                "mrr": {
                    "mock": mock_data["metrics"]["mrr_final"],
                    "real": real_data["metrics"]["mrr_final"]
                },
                "ndcg": {
                    "mock": mock_data["metrics"]["ndcg_final"],
                    "real": real_data["metrics"]["ndcg_final"]
                },
                "precision": {
                    "mock": mock_data["metrics"]["precision_final"],
                    "real": real_data["metrics"]["precision_final"]
                },
                "avg_time": {
                    "mock": mock_data["avg_time_per_query"],
                    "real": real_data["avg_time_per_query"]
                }
            }
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """保存实验结果"""
        if filename is None:
            filename = f"v6_experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        out_dir = results_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"实验结果已保存到: {filepath}")
        return filepath
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """运行完整V6实验"""
        
        # 设置实验
        self.setup_experiment()
        
        # 运行Mock实验
        mock_data = self.run_mock_experiment()
        
        # 运行真实模型实验
        real_data = self.run_real_experiment()
        
        # 生成对比报告
        comparison_report = self.generate_comparison_report(mock_data, real_data)
        
        # 整合结果
        final_results = {
            "experiment_info": {
                "name": self.experiment_name,
                "timestamp": self.timestamp,
                "document_count": len(self.documents),
                "query_count": len(self.test_cases)
            },
            "mock_results": mock_data,
            "real_results": real_data,
            "comparison": comparison_report
        }
        
        # 保存结果
        saved_file = self.save_results(final_results)
        
        # 更新研究历史
        self.update_research_history(final_results)
        
        return final_results
    
    def update_research_history(self, results: Dict[str, Any]):
        """更新研究历史"""
        history_path = data_dir() / "research_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = {"iterations": [], "plan": []}
        
        # 添加V6实验结果
        v6_entry = {
            "id": len(history.get("iterations", [])) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "V6: Real Model Integration & Performance Validation",
            "metrics": {
                "mrr_retrieval": results["real_results"]["metrics"]["mrr_retrieval"],
                "mrr_final": results["real_results"]["metrics"]["mrr_final"],
                "ndcg_retrieval": results["real_results"]["metrics"]["ndcg_retrieval"],
                "ndcg_final": results["real_results"]["metrics"]["ndcg_final"]
            },
            "comparison": results["comparison"]["performance_comparison"],
            "correlation": results["comparison"]["correlation_analysis"]
        }
        
        history["iterations"].append(v6_entry)
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info("研究历史已更新")
    
    def print_summary(self, results: Dict[str, Any]):
        """打印实验摘要"""
        print("\n" + "="*80)
        print("V6实验结果摘要")
        print("="*80)
        
        comparison = results["comparison"]
        
        print(f"模型相关性:")
        print(f"  Pearson相关系数: {comparison['correlation_analysis']['pearson']:.3f}")
        print(f"  Spearman相关系数: {comparison['correlation_analysis']['spearman']:.3f}")
        
        print(f"\n性能提升:")
        print(f"  MRR提升: {comparison['performance_comparison']['mrr_improvement']:.1f}%")
        print(f"  NDCG提升: {comparison['performance_comparison']['ndcg_improvement']:.1f}%")
        print(f"  时间开销: {comparison['performance_comparison']['time_overhead_pct']:.1f}% ({comparison['performance_comparison']['time_overhead_s']:.3f}s)")
        
        print(f"\n详细指标:")
        mock_vs_real = comparison["mock_vs_real"]
        print(f"  MRR - Mock: {mock_vs_real['mrr']['mock']:.3f}, Real: {mock_vs_real['mrr']['real']:.3f}")
        print(f"  NDCG - Mock: {mock_vs_real['ndcg']['mock']:.3f}, Real: {mock_vs_real['ndcg']['real']:.3f}")
        print(f"  时间 - Mock: {mock_vs_real['avg_time']['mock']:.3f}s, Real: {mock_vs_real['avg_time']['real']:.3f}s")


def main():
    """主函数"""
    experiment = V6Experiment()
    
    try:
        results = experiment.run_full_experiment()
        experiment.print_summary(results)
        
        logger.info("\n✅ V6实验完成！")
        logger.info("结果已保存到 research_history.json")
        
    except Exception as e:
        logger.error(f"实验执行失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()
