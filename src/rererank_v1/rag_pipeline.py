import numpy as np
from typing import List, Dict, Optional, Any
import re
import sys
import logging
from collections import Counter
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing the required libraries
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
except ImportError:
    logger.error("Required libraries not found.")
    logger.error("Please install: pip install -r requirements.txt")
    sys.exit(1)

# Import V6 real reranker
try:
    from .real_reranker import RealReranker, MockReranker, RerankerFactory
    V6_AVAILABLE = True
except ImportError:
    logger.warning("V6 real reranker not available, using legacy mock")
    V6_AVAILABLE = False

from .evidence_chain import build_reasoning_graph, format_chain_for_llm
from .cove_verifier import CoVeVerifier

class RAGPipeline:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', 
                 reranker_model_name: str = 'BAAI/bge-reranker-base',
                 device: str = None, use_v6_reranker: bool = False):
        """
        Initialize the RAG pipeline with embedding and reranker models.
        
        Args:
            embedding_model_name: Name of the embedding model
            reranker_model_name: Name of the reranker model
            device: Device to run models on ('cpu', 'cuda', 'mps', or None for auto-detection)
            use_v6_reranker: Whether to use V6 real reranker implementation
        """
        # Check for forced mock mode via environment variable
        self.mock_mode = os.getenv('FORCE_MOCK', '0') == '1'
        self.use_v6_reranker = use_v6_reranker and V6_AVAILABLE
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        logger.info(f"Using device: {self.device}")
        logger.info(f"V6 Reranker Mode: {self.use_v6_reranker}")
        
        if self.mock_mode:
            logger.info("🔧 FORCE_MOCK enabled. Skipping model loading and using Mock Mode.")
            self.encoder = self._create_mock_encoder()
            if self.use_v6_reranker:
                self.reranker = RerankerFactory.create_reranker("mock")
            else:
                self.reranker = self._create_mock_reranker()
        else:
            logger.info(f"Loading Embedding Model: {embedding_model_name}...")
            try:
                self.encoder = SentenceTransformer(embedding_model_name, device=device)
            except Exception as e:
                logger.warning(f"Failed to load Embedding Model due to: {e}")
                logger.warning("Switching to MOCK MODE for Embedding Model.")
                self.mock_mode = True
                self.encoder = self._create_mock_encoder()

            if self.use_v6_reranker:
                logger.info("Loading V6 Real Reranker...")
                try:
                    self.reranker = RerankerFactory.create_reranker("real", device=device)
                except Exception as e:
                    logger.warning(f"Failed to load V6 Real Reranker due to: {e}")
                    logger.warning("Falling back to legacy reranker.")
                    self.use_v6_reranker = False
                    try:
                        self.reranker = CrossEncoder(reranker_model_name, device=device)
                    except Exception as e2:
                        logger.warning(f"Failed to load legacy reranker due to: {e2}")
                        logger.warning("Switching to MOCK MODE for Reranker Model.")
                        self.mock_mode = True
                        self.reranker = self._create_mock_reranker()
            else:
                logger.info(f"Loading Reranker Model: {reranker_model_name}...")
                try:
                    self.reranker = CrossEncoder(reranker_model_name, device=device)
                except Exception as e:
                    logger.warning(f"Failed to load Reranker Model due to: {e}")
                    logger.warning("Switching to MOCK MODE for Reranker Model.")
                    self.mock_mode = True
                    self.reranker = self._create_mock_reranker()
        
        self.documents = [] # Used for legacy text corpus
        self.evidence_units = [] # Used for heterogeneous Phase 2
        self.doc_embeddings = None
        self.stats = {
            'total_tokens': 0,
            'total_latency': 0.0,
            'retrieval_calls': 0,
            'reranker_calls': 0
        }

    def _count_tokens(self, text: str) -> int:
        # Simple whitespace-based token approximation
        return len(text.split())

    def _create_mock_encoder(self):
        """Creates a mock encoder for testing without model weights."""
        class MockEncoder:
            def __init__(self, model_name):
                self.model_name = model_name

            def encode(self, texts, convert_to_tensor=False, **kwargs):
                # Helper to generate a consistent vector based on hash of text
                def get_vec(text):
                    text_lower = text.lower()
                    np.random.seed(hash(text_lower) % 2**32)
                    vec = np.random.rand(384)
                    
                    # Add stronger signal for keyword matches to make retrieval non-random
                    topics = {
                        "machine learning": 0, "ml": 0, "supervised": 0, "deep learning": 0,
                        "natural language": 1, "nlp": 1, "transformer": 1, "bert": 1,
                        "information retrieval": 2, "search": 2, "rerank": 2, "rrf": 2,
                        "python": 3, "pandas": 3, "numpy": 3, "scikit": 3,
                        "java": 4, "apple": 5, "amazon": 6, "bank": 7
                    }
                    
                    for topic, idx in topics.items():
                        if topic in text_lower:
                            vec[idx*10:(idx+1)*10] += 5.0
                    
                    # Normalize
                    return vec / np.linalg.norm(vec)
                
                if isinstance(texts, str):
                    return [get_vec(texts)]
                return [get_vec(t) for t in texts]
                
        return MockEncoder(self.encoder.model_name_or_path if hasattr(self, 'encoder') and hasattr(self.encoder, 'model_name_or_path') else 'mock-model')

    def _create_mock_reranker(self):
        """Creates a mock reranker for testing without model weights."""
        class MockReranker:
            def predict(self, pairs):
                scores = []
                for q, d in pairs:
                    # Semantic overlap simulation
                    q_words = set(re.findall(r'\w+', q.lower()))
                    d_words = set(re.findall(r'\w+', d.lower()))
                    overlap = len(q_words.intersection(d_words))
                    
                    # Base score from overlap
                    score = (overlap / len(q_words)) * 5.0 if q_words else 0
                    
                    # Add noise
                    score += np.random.normal(0, 0.5)
                    
                    # Boost specific "strong" matches (simulation)
                    q_lower = q.lower()
                    d_lower = d.lower()
                    
                    if "bert" in q_lower and "bert" in d_lower: score += 3.0
                    if "python" in q_lower and "python" in d_lower: score += 2.0
                    
                    # Handle Ambiguity Logic (Context Awareness Simulation)
                    # Python: Language vs Snake
                    if "python" in q_lower and "language" in q_lower:
                        if "snake" in d_lower: score -= 5.0 # Penalize wrong context
                        if "programming" in d_lower: score += 3.0
                    
                    # Amazon: River vs Company
                    if "amazon" in q_lower:
                        if "river" in q_lower and "company" in d_lower: score -= 5.0
                        if "river" in q_lower and "water" in d_lower: score += 3.0
                    
                    # Apple: Tech vs Fruit
                    if "apple" in q_lower:
                        if "technology" in q_lower and "fruit" in d_lower: score -= 5.0
                        if "technology" in q_lower and "iphone" in d_lower: score += 3.0

                    # Handle Negation (Simulation for V3)
                    if "not" in q_lower:
                        # E.g. "not named python" -> penalize python docs
                        if "python" in q_lower and "python" in d_lower:
                            score -= 8.0 # Strong penalty for negated term
                            
                    # Ensemble/Fusion Boost (Simulating Multi-View)
                    # If document is long (more context), give slight boost
                    if len(d) > 50: score += 0.5

                    scores.append(score)
                return np.array(scores)
        return MockReranker()

    def add_documents(self, documents: List[str]):
        """
        Add documents to the knowledge base and encode them. (Legacy text-only)
        """
        self.documents = documents
        logger.info(f"Encoding {len(documents)} documents...")
        
        embeddings = self.encoder.encode(documents, convert_to_tensor=False)
        
        # Handle different return types/shapes from encoder/mock
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
            
        self.doc_embeddings = embeddings
        
        # Normalize embeddings for cosine similarity
        norm = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        self.doc_embeddings = self.doc_embeddings / (norm + 1e-10)

    def add_evidence_units(self, evidence_units: list):
        """
        Phase 2: Add Heterogeneous Knowledge (EvidenceUnit)
        """
        self.evidence_units = evidence_units
        logger.info(f"Encoding {len(evidence_units)} heterogeneous evidence units...")
        
        # Extract content for embedding
        contents = [eu.content for eu in evidence_units]
        embeddings = self.encoder.encode(contents, convert_to_tensor=False)
        
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
            
        self.doc_embeddings = embeddings
        norm = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        self.doc_embeddings = self.doc_embeddings / (norm + 1e-10)

    def _retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        First-stage retrieval using vector similarity (Cosine Similarity).
        """
        import time
        start_t = time.time()
        self.stats['retrieval_calls'] += 1
        self.stats['total_tokens'] += self._count_tokens(query)

        query_embedding = self.encoder.encode(query, convert_to_tensor=False)
        
        # Ensure query embedding is 2D array (1, dim)
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / (query_norm + 1e-10)
        
        # Calculate cosine similarity
        similarities = np.dot(self.doc_embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if self.evidence_units:
                # Phase 2: Heterogeneous Retrieval
                eu = self.evidence_units[idx]
                text_content = eu.content
                source = eu.source
            else:
                # Phase 1 / Legacy
                text_content = self.documents[idx]
                source = "text"
                
            results.append({
                'id': int(idx),
                'text': text_content,
                'source': source,
                'metadata': eu.metadata if self.evidence_units else {},
                'score': float(similarities[idx]),
                'rank': 0  # To be filled later
            })
            self.stats['total_tokens'] += self._count_tokens(text_content)
        
        # Assign ranks (1-based)
        for i, res in enumerate(results):
            res['rank'] = i + 1
            
        self.stats['total_latency'] += (time.time() - start_t)
        return results

    def _rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Second-stage reranking using Cross-Encoder or V6 Real Reranker.
        In Phase 2, injects source type into the text for explicit awareness.
        """
        if not candidates:
            return []
            
        import time
        start_t = time.time()
        self.stats['reranker_calls'] += 1
        self.stats['total_tokens'] += self._count_tokens(query) * len(candidates)
        
        # Inject source explicitly for Phase 2: "[source=table] Row: ..."
        formatted_texts = [
            f"(source={doc.get('source', 'text')}) {doc['text']}" if 'source' in doc else doc['text'] 
            for doc in candidates
        ]
        pairs = [[query, text] for text in formatted_texts]
        
        if self.use_v6_reranker:
            # Use V6 real reranker
            scores = self.reranker.score(query, formatted_texts)
            # V6 reranker already returns probabilities [0,1]
            probs = scores
        else:
            # Use legacy CrossEncoder
            scores = self.reranker.predict(pairs)
            # Normalize scores using sigmoid to get probability-like score [0, 1]
            def sigmoid(x):
                return 1 / (1 + np.exp(-np.array(x)))
            probs = sigmoid(scores)
        
        # Update scores and sort
        for i, doc in enumerate(candidates):
            doc['rerank_score'] = float(probs[i])
            
        # Sort by reranker score
        reranked_results = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        # Update ranks after reranking
        for i, res in enumerate(reranked_results):
            res['rank'] = i + 1
            
        self.stats['total_latency'] += (time.time() - start_t)
        return reranked_results

    def _extract_keywords(self, text: str, top_n: int = 5) -> str:
        """
        Simple keyword extraction for Pseudo-Relevance Feedback.
        """
        # Basic stopword list
        stopwords = set(['the', 'is', 'at', 'which', 'on', 'in', 'a', 'an', 'and', 'or', 'for', 'of', 'to', 'with', 'as', 'by', 'that', 'this', 'it'])
        
        words = re.findall(r'\w+', text.lower())
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        counter = Counter(filtered_words)
        most_common = counter.most_common(top_n)
        keywords = [word for word, count in most_common]
        
        return " ".join(keywords)

    def _rrf_fusion(self, list1: List[Dict], list2: List[Dict], k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) to combine two ranked lists.
        Score = sum(1 / (k + rank_i))
        """
        fusion_scores = {}
        
        # Helper to process a list
        def process_list(doc_list, source_name):
            for doc in doc_list:
                doc_id = doc['id']
                if doc_id not in fusion_scores:
                    fusion_scores[doc_id] = {
                        'text': doc['text'], 
                        'source': doc.get('source', 'text'),
                        'metadata': doc.get('metadata', {}),
                        'score': 0.0, 
                        'sources': [],
                        'original_scores': {}
                    }
                fusion_scores[doc_id]['score'] += 1.0 / (k + doc['rank'])
                fusion_scores[doc_id]['sources'].append(source_name)
                fusion_scores[doc_id]['original_scores'][source_name] = doc.get('rerank_score', 0.0)

        process_list(list1, 'original_query')
        process_list(list2, 'expanded_query')
            
        # Convert to list and sort
        fused_results = []
        for doc_id, info in fusion_scores.items():
            fused_results.append({
                'id': doc_id,
                'text': info['text'],
                'source': info['source'],
                'metadata': info['metadata'],
                'rrf_score': info['score'],
                'sources': info['sources'],
                'details': info['original_scores']
            })
            
        return sorted(fused_results, key=lambda x: x['rrf_score'], reverse=True)

    def _judge_need_more_context(self, query: str, top_score: float) -> bool:
        """
        Mock LLM-based Self-RAG judgment: Does the current top result sufficiently answer the query?
        """
        query_lower = query.lower()
        # Multi-hop indicators
        multi_hop_triggers = ["who", "which", "where", "when", "based on", "award"]
        
        # If it's a known multi-hop query structure or confidence is borderline
        is_multi_hop = any(trigger in query_lower for trigger in multi_hop_triggers)
        
        if is_multi_hop and top_score < 0.95: 
            return True
        if top_score > 0.6 and top_score < 0.85: # Borderline confidence
            return True
        return False

    def search(self, query: str, top_k: int = 5, prf_threshold: float = 0.8, active_retrieval: bool = True) -> List[Dict]:
        """
        Main RAG workflow:
        1. Retrieve (Vector Search)
        2. Rerank (BGE-Reranker)
        3. Conditional PRF / Active Retrieval
        4. Fusion (RRF)
        """
        import time
        start_t = time.time()
        logger.info(f"Processing Query: {query}")
        
        # Step 1: Initial Retrieval
        candidates_1 = self._retrieve(query, top_k=top_k)
        logger.info(f"Step 1: Retrieved {len(candidates_1)} docs.")
        
        # Step 2: Reranking
        reranked_1 = self._rerank(query, candidates_1)
        top_score = reranked_1[0]['rerank_score'] if reranked_1 else 0.0
        logger.info(f"Step 2: Top Reranker Score: {top_score:.4f}")

        query_lower = query.lower()
        adaptive_threshold = prf_threshold
        if "not" in query_lower or "except" in query_lower:
            adaptive_threshold = max(prf_threshold, 0.9)
        elif "larger than" in query_lower or "smaller than" in query_lower:
            adaptive_threshold = max(prf_threshold, 0.88)
        elif "and" in query_lower:
            adaptive_threshold = max(prf_threshold, 0.85)
        
        # Step 3: Check PRF / Active Retrieval Condition
        needs_more_context = False
        if active_retrieval:
            needs_more_context = self._judge_need_more_context(query, top_score)
            if needs_more_context:
                logger.info(">>> [Active Retrieval] LLM-judge indicates missing context for multi-hop. Triggering secondary retrieval...")
        
        if not needs_more_context and top_score > adaptive_threshold:
            needs_more_context = True
            logger.info(">>> High confidence detected. Triggering Pseudo-Relevance Feedback (PRF)...")
            
        if needs_more_context:
            # Extract keywords from top document
            top_doc_text = reranked_1[0]['text']
            keywords = self._extract_keywords(top_doc_text)
            new_query = f"{query} {keywords}"
            logger.info(f"Generated New Query: {new_query}")
            
            # Step 4: Second Retrieval with new query
            candidates_2 = self._retrieve(new_query, top_k=top_k)
            # Rerank second list too
            reranked_2 = self._rerank(new_query, candidates_2)
            
            # Step 5: Fusion
            logger.info("Step 5: Fusing results with RRF...")
            final_results = self._rrf_fusion(reranked_1, reranked_2)
            
            self.stats['total_latency'] += (time.time() - start_t)
            return final_results
        else:
            logger.info(">>> Context sufficient or confidence below threshold. Returning initial reranked results.")
            self.stats['total_latency'] += (time.time() - start_t)
            return reranked_1

    def search_with_chain(self, query: str, top_k: int = 5, prf_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Phase 3 feature: Executes RAG search and constructs an Evidence Chain (Graph of Thoughts).
        """
        logger.info(f"Phase 3: Executing Search with Evidence Chain for: '{query}'")
        
        # Get standard RAG results
        results = self.search(query, top_k=top_k, prf_threshold=prf_threshold)
        
        # Build Reasoning Graph / Evidence Chain
        chain_roots = build_reasoning_graph(results, similarity_threshold=0.01) # Lower threshold for demo
        chain_str = format_chain_for_llm(chain_roots)
        
        return {
            "results": results,
            "chain": [node.to_dict() for node in chain_roots],
            "chain_str": chain_str
        }

    def verify_answer(
        self,
        generated_answer: str,
        evidence_chain: List[Dict[str, Any]],
        confidence_threshold: float = 0.5,
        backend: str = "heuristic",
        model: str = "deepseek-v4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        decision_policy: str = "soft",
        min_claim_confidence: float | None = None,
    ) -> Dict[str, Any]:
        """
        Phase 4 feature: Chain-of-Verification (CoVe).
        Evaluates a generated answer against the retrieved evidence chain.
        Returns ACCEPTED or REJECTED (No-Answer).
        """
        logger.info(f"Phase 4: Running CoVe Verification on Answer: '{generated_answer}'")
        verifier = CoVeVerifier(
            confidence_threshold=confidence_threshold,
            backend=backend,
            model=model,
            api_key=api_key,
            base_url=base_url,
            decision_policy=decision_policy,
            min_claim_confidence=min_claim_confidence,
        )
        
        cove_result = verifier.evaluate_answer(generated_answer, evidence_chain)
        logger.info(f"CoVe Status: {cove_result['status']} | Reason: {cove_result['reason']}")
        return cove_result

def heuristic_generate_answer(query: str, results) -> str:
    """
    Enhanced Heuristic Generation using Knowledge Graphs and Table Rows.
    Instead of blind string overlap, it seeks exact entity bridging.
    """
    if not results:
        return "No-Answer"

    # Extract all nouns/entities from query (simplified)
    q_words = set(re.findall(r"\b[A-Z][a-z]+\b", query))
    
    candidates = []
    
    for doc in results[:4]:
        text = doc.get("text", "")
        source = doc.get("source", "text")
        
        # Prioritize structured knowledge: Table & Graph
        if source == "graph":
            # Graph Fact: A -> relation -> B
            if "->" in text:
                parts = text.split("->")
                if len(parts) >= 3:
                    ans_candidate = parts[-1].strip()
                    # If query mentions the subject, the object is likely the answer
                    if any(w in parts[0] for w in q_words):
                        candidates.append((5.0, ans_candidate))
        
        elif source == "table":
            # Table: Title | Key: Val
            if "|" in text and ":" in text:
                parts = text.split("|")[-1].split(":")
                if len(parts) >= 2:
                    ans_candidate = parts[-1].strip()
                    candidates.append((4.0, ans_candidate))
                    
        # Fallback to Text sentence overlap
        for sent in re.split(r"[.!?。；;]", text):
            sent = sent.strip()
            if len(sent) < 8:
                continue
            s_tokens = set(re.findall(r"\w+", sent.lower()))
            q_tokens = set(re.findall(r"\w+", query.lower()))
            overlap = len(q_tokens & s_tokens)
            if overlap > 0:
                candidates.append((overlap * 0.5, sent))

    if not candidates:
        return results[0].get("text", "")[:120]

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]
    return best[:200]

# Example Usage
if __name__ == "__main__":
    # Create a dummy corpus
    corpus = [
        "Python is a programming language that lets you work quickly.",
        "Java is a high-level, class-based, object-oriented programming language.",
        "The BGE Reranker is a powerful model for text retrieval.",
        "Sentence Transformers provides easy methods to compute dense vector representations.",
        "Pseudo-relevance feedback is a technique used in information retrieval.",
        "Reciprocal Rank Fusion (RRF) is a method to combine multiple search results.",
        "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'.",
    ]
    
    # Initialize pipeline
    rag = RAGPipeline()
    rag.add_documents(corpus)
    
    # Test Case 1: Query that should trigger PRF (High relevance expected)
    query1 = "What is BGE Reranker?"
    results1 = rag.search(query1, top_k=3, prf_threshold=0.5) 
    
    print("\nFinal Results for Query 1:")
    for i, res in enumerate(results1[:3]):
        score = res.get('rrf_score', res.get('rerank_score'))
        print(f"{i+1}. {res['text']} (Score: {score:.4f})")

    # Test Case 2: Ambiguous query 
    query2 = "food recipes"
    results2 = rag.search(query2, top_k=3)
    
    print("\nFinal Results for Query 2:")
    for i, res in enumerate(results2[:3]):
        score = res.get('rrf_score', res.get('rerank_score'))
        print(f"{i+1}. {res['text']} (Score: {score:.4f})")
