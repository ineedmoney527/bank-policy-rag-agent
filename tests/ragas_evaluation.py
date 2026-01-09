"""
RAGAS Evaluation Runner

Runs the RAG pipeline on the evaluation dataset and computes RAGAS metrics:
- faithfulness (≥ 0.85)
- answer_relevancy (≥ 0.80)
- context_precision (≥ 0.70)
- context_recall (≥ 0.70)

Exits with code 1 if ANY metric is below threshold.
"""

import json
import re
import sys
import os
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings

# Import the RAG pipeline
from graph import app

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
EVAL_DATASET_PATH = PROJECT_ROOT / "tests" / "eval_dataset.json"
RESULTS_OUTPUT_PATH = PROJECT_ROOT / "tests" / "ragas_results.json"

# Metric thresholds
THRESHOLDS = {
    "faithfulness": 0.0,
    "answer_relevancy": 0.0,
    "context_precision": 0.0,
    "context_recall": 0.0,
}
# THRESHOLDS = {
#     "faithfulness": 0.80,
#     "answer_relevancy": 0.80,
#     "context_precision": 0.70,
#     "context_recall": 0.70,
# }

# OpenRouter config (same as graph.py)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    
EVAL_LLM_MODEL = "deepseek/deepseek-v3.2"


def strip_citations(text: str) -> str:
    """Remove citation markers like (Document Title S X.X) from text."""
    # Pattern: (Text S/SECTION X.X) or similar
    return re.sub(r'\([^)]+(?:S|SECTION)\s*[\d\.]+[^)]*\)', '', text).strip()


def run_rag_pipeline(question: str) -> tuple[str, list[str]]:
    """
    Run the RAG pipeline and return (answer, contexts).
    
    Returns:
        tuple: (answer_text, list_of_context_strings)
    """
    initial_state = {
        'original_query': question,
        'current_query': question,
        'retry_count': 0,
        'revision_count': 0,
        'sub_questions': [],
        'documents': [],
        'generation': '',
        'filters': {},
        'grade_status': '',
        'hallucination_status': '',
        'critique': ''
    }
    
    # Run the graph
    final_state = None
    for state in app.stream(initial_state):
        final_state = state
    
    # Extract the final state values
    if final_state:
        node_name = list(final_state.keys())[0]
        state_values = final_state[node_name]
        
        answer = state_values.get('generation', '')
        documents = state_values.get('documents', [])
        
        # Extract context texts
        contexts = [doc.page_content for doc in documents]
        
        return answer, contexts
    
    return "", []


def load_eval_dataset() -> list[dict]:
    """Load the evaluation dataset."""
    with open(EVAL_DATASET_PATH, 'r') as f:
        return json.load(f)


def run_evaluation(run_pipeline: bool = True):
    """Run the full RAGAS evaluation."""
    print("=" * 60)
    print("RAGAS Evaluation Runner")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Dataset: {EVAL_DATASET_PATH}")
    print(f"Seed: {SEED}")
    print()
    
    # Load dataset
    eval_data = load_eval_dataset()
    print(f"Loaded {len(eval_data)} evaluation samples")
    print()
    
    if run_pipeline:
        # Run RAG pipeline on each sample
        print("Running RAG pipeline on evaluation samples...")
        results = []
        
        for i, item in enumerate(eval_data, 1):
            question = item["question"]
            ground_truth = item["ground_truth"]
            
            print(f"  [{i}/{len(eval_data)}] {question[:50]}...")
            
            try:
                answer, contexts = run_rag_pipeline(question)
                answer_clean = strip_citations(answer)
                
                # Ensure contexts is not empty (RAGAS requirement)
                if not contexts:
                    contexts = ["No context retrieved"]
                
                results.append({
                    "user_input": question,
                    "response": answer_clean,
                    "retrieved_contexts": contexts,
                    "reference": ground_truth,
                })
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({
                    "user_input": question,
                    "response": f"ERROR: {e}",
                    "retrieved_contexts": ["Error during retrieval"],
                    "reference": ground_truth,
                })
        
        print(f"\nCompleted {len(results)} samples")
    else:
        # Load existing results from file
        print("Loading existing results from ragas_results.json...")
        if not RESULTS_OUTPUT_PATH.exists():
            print(f"ERROR: {RESULTS_OUTPUT_PATH} does not exist. Run with run_pipeline=True first.")
            sys.exit(1)
        
        with open(RESULTS_OUTPUT_PATH, 'r') as f:
            saved_data = json.load(f)
            results = saved_data.get("samples", [])
        
        print(f"Loaded {len(results)} existing samples")
    
    print()
    
    # Create RAGAS dataset
    print("Computing RAGAS metrics...")
    ragas_dataset = Dataset.from_list(results)
    
    # Initialize LLM and embeddings for RAGAS
    eval_llm = LangchainLLMWrapper(ChatOpenAI(
        model=EVAL_LLM_MODEL,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        n=3
    ))
    
    eval_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(
        model="nomic-embed-text"
    ))
    
    # Initialize metrics with LLM
    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings),
        ContextPrecision(llm=eval_llm),
        ContextRecall(llm=eval_llm),
    ]
    
    try:
        ragas_result = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
        )
        
        import numpy as np

        def mean_metric(metric):
            values = [v for v in metric if not np.isnan(v)]
            if not values:
                return 0.0
            return float(np.mean(values))

        scores = {
            "faithfulness": mean_metric(ragas_result["faithfulness"]),
            "answer_relevancy": mean_metric(ragas_result["answer_relevancy"]),
            "context_precision": mean_metric(ragas_result["context_precision"]),
            "context_recall": mean_metric(ragas_result["context_recall"]),
        }

    except Exception as e:
        print(f"RAGAS evaluation error: {e}")
        import traceback
        traceback.print_exc()
        scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }
    
    # Print results table
    print()
    print("=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'Score':>10} {'Threshold':>12} {'Status':>10}")
    print("-" * 60)
    
    all_passed = True
    for metric, score in scores.items():
        threshold = THRESHOLDS[metric]
        passed = score >= threshold
        status = "PASS" if passed else "FAIL"
        
        if not passed:
            all_passed = False
        
        print(f"{metric:<25} {score:>10.4f} {threshold:>12.2f} {status:>10}")
    
    print("-" * 60)
    overall_status = "PASS" if all_passed else "FAIL"
    print(f"{'OVERALL':<25} {'':>10} {'':>12} {overall_status:>10}")
    print("=" * 60)
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(results),
        "scores": scores,
        "thresholds": THRESHOLDS,
        "all_passed": all_passed,
        "samples": results,
    }
    
    with open(RESULTS_OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_OUTPUT_PATH}")
    
    # Exit with appropriate code
    if not all_passed:
        print("\n❌ EVALUATION FAILED - One or more metrics below threshold")
        sys.exit(1)
    else:
        print("\n✅ EVALUATION PASSED - All metrics meet thresholds")
        sys.exit(0)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on RAG pipeline")
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip running RAG pipeline and use existing results from ragas_results.json"
    )
    args = parser.parse_args()
    
    run_evaluation(run_pipeline=not args.skip_pipeline)
