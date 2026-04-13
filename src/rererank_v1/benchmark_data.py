from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TestCase:
    query: str
    relevant_doc_ids: List[int]
    description: str

# Define a more realistic dataset for research
DOCUMENTS = [
    # Topic 1: Machine Learning & AI
    "Machine learning is a subset of artificial intelligence.",  # 0
    "Deep learning uses neural networks with many layers.",      # 1
    "Supervised learning requires labeled training data.",       # 2
    "Unsupervised learning finds patterns in unlabeled data.",   # 3
    "Reinforcement learning is about agents taking actions.",    # 4
    
    # Topic 2: Natural Language Processing (NLP)
    "Natural Language Processing (NLP) enables computers to understand text.", # 5
    "Transformers are state-of-the-art models for NLP tasks.",               # 6
    "BERT is a bidirectional transformer model from Google.",                # 7
    "GPT-3 is a large language model developed by OpenAI.",                  # 8
    "Tokenization splits text into smaller units called tokens.",            # 9

    # Topic 3: Information Retrieval (IR) & Search
    "Information retrieval is the science of searching for information.",    # 10
    "Vector search uses embeddings to find semantic similarity.",            # 11
    "A reranker reorders search results to improve relevance.",              # 12
    "Reciprocal Rank Fusion (RRF) combines multiple ranked lists.",          # 13
    "Pseudo-relevance feedback expands queries using top results.",          # 14

    # Topic 4: Python Programming
    "Python is a popular programming language for data science.",            # 15
    "Pandas is a Python library for data manipulation.",                     # 16
    "NumPy provides support for large, multi-dimensional arrays.",           # 17
    "Scikit-learn is a machine learning library for Python.",                # 18
    "PyTorch is an open source machine learning framework.",                 # 19

    # Topic 5: Distractors / Ambiguous Concepts (for Robustness Testing)
    "Java is a popular programming language but different from Python.",     # 20
    "A python is a large non-venomous snake found in Africa and Asia.",      # 21 (Ambiguity)
    "Apple creates hardware like iPhone and software like iOS.",             # 22
    "An apple a day keeps the doctor away.",                                 # 23 (Ambiguity with brand)
    "Amazon is a large technology company focusing on e-commerce.",          # 24
    "The Amazon river is the largest river by discharge volume.",            # 25 (Ambiguity)
    "Bank of America is a multinational investment bank.",                   # 26
    "A river bank is the land alongside a body of water.",                   # 27 (Ambiguity)

    # Topic 7: Scalability Filler (General Tech) - Iteration V4
    "Cloud computing provides on-demand computing system resources.",        # 28
    "AWS is a comprehensive cloud platform from Amazon.",                    # 29
    "Azure is a cloud computing service created by Microsoft.",              # 30
    "Google Cloud Platform provides infrastructure as a service.",           # 31
    "Docker is a set of platform as a service products using OS-level virtualization.", # 32
    "Kubernetes is an open-source container orchestration system.",          # 33
    "DevOps is a set of practices that combines software development and IT operations.", # 34
    "CI/CD bridges the gaps between development and operation activities.",  # 35
    "Git is a distributed version control system.",                          # 36
    "GitHub is a provider of Internet hosting for software development.",    # 37
    "Agile software development advocates adaptive planning.",               # 38
    "Scrum is a framework for developing, delivering, and sustaining products.", # 39
    "Jira is a proprietary issue tracking product developed by Atlassian.",  # 40
    "Confluence is a web-based corporate wiki developed by Atlassian.",      # 41
    "Slack is a proprietary business communication platform.",               # 42
    "Zoom is a videotelephony software program developed by Zoom Video Communications.", # 43
    "Microsoft Teams is a business communication platform.",                 # 44
    "Visual Studio Code is a source-code editor made by Microsoft.",         # 45
    "IntelliJ IDEA is an integrated development environment written in Java.", # 46
    "PyCharm is an integrated development environment used in computer programming.", # 47

    # Topic 8: Multi-Hop Reasoning (HotpotQA Style) - Phase 1
    "Arthur C. Clarke wrote the science fiction novel '2001: A Space Odyssey'.", # 48
    "Stanley Kubrick directed the film adaptation of '2001: A Space Odyssey'.", # 49
    "The Turing Award is often referred to as the 'Nobel Prize of Computing'.", # 50
    "Geoffrey Hinton, Yann LeCun, and Yoshua Bengio won the Turing Award in 2018.", # 51
    "Geoffrey Hinton works at the University of Toronto.", # 52
    "Yoshua Bengio is a professor at the University of Montreal.", # 53
    "Yann LeCun is the Chief AI Scientist at Meta (formerly Facebook).", # 54
]

TEST_CASES = [
    TestCase(
        query="What is machine learning?",
        relevant_doc_ids=[0, 1, 2, 3, 4, 18, 19],
        description="General ML definition"
    ),
    TestCase(
        query="Tell me about NLP models like BERT",
        relevant_doc_ids=[5, 6, 7, 8],
        description="Specific NLP models"
    ),
    TestCase(
        query="How does search ranking work?",
        relevant_doc_ids=[10, 11, 12, 13, 14],
        description="Search ranking concepts"
    ),
    TestCase(
        query="Python libraries for data",
        relevant_doc_ids=[15, 16, 17, 18],
        description="Python ecosystem"
    ),
    TestCase(
        query="Explain reinforcement learning",
        relevant_doc_ids=[4],
        description="Specific ML type"
    ),
    # New Challenging Cases
    TestCase(
        query="python programming language",
        relevant_doc_ids=[15, 16, 17, 18, 19], # Should exclude snake (21)
        description="Ambiguous Term: Python (Language vs Snake)"
    ),
    TestCase(
        query="amazon river location",
        relevant_doc_ids=[25], # Should exclude company (24)
        description="Ambiguous Term: Amazon (River vs Company)"
    ),
    TestCase(
        query="financial bank services",
        relevant_doc_ids=[26], # Should exclude river bank (27)
        description="Ambiguous Term: Bank (Finance vs River)"
    ),
    TestCase(
        query="apple technology products",
        relevant_doc_ids=[22], # Should exclude fruit (23)
        description="Ambiguous Term: Apple (Tech vs Fruit)"
    ),

    # Topic 6: Complex Queries (Negation & Comparison) - Iteration V3
    TestCase(
        query="programming languages not named python",
        relevant_doc_ids=[20], # Java (20). Should exclude Python (15)
        description="Negation Query"
    ),
    TestCase(
        query="river flows larger than amazon",
        relevant_doc_ids=[], # No document matches this factually, but should definitely NOT retrieve Amazon company (24)
        description="Comparison/Attribute Query"
    ),

    # Topic 7: Broad Tech Query - Iteration V4
    TestCase(
        query="cloud platforms and container tools",
        relevant_doc_ids=[28, 29, 30, 31, 32, 33],
        description="Multi-concept Query"
    ),
    TestCase(
        query="python data tools but not deep learning frameworks",
        relevant_doc_ids=[16, 17],
        description="Constrained Query"
    ),
    TestCase(
        query="quantum blockchain healthcare compliance",
        relevant_doc_ids=[],
        description="No-Answer Safety Query"
    ),

    # Phase 1: Multi-Hop Queries
    TestCase(
        query="Who directed the movie based on the book by Arthur C. Clarke?",
        relevant_doc_ids=[48, 49],
        description="Multi-hop: Author -> Book -> Director"
    ),
    TestCase(
        query="Which 2018 Turing Award winner works at Meta?",
        relevant_doc_ids=[51, 54],
        description="Multi-hop: Turing Award -> Winner -> Company"
    ),
    TestCase(
        query="What award is known as the Nobel Prize of Computing and who won it in 2018?",
        relevant_doc_ids=[50, 51],
        description="Multi-hop: Award Name -> Winners"
    )
]
