"""
Quick test script for the LangGraph workflow to verify it works end-to-end
"""

import sys
sys.path.insert(0, '/Users/choojunheng/.gemini/antigravity/scratch/agent/src')

from graph import app

# Simple test query that shouldn't require sub-questions
query = "What is the minimum capital requirement for banks?"

print("=" * 60)
print("Testing LangGraph RAG Workflow")
print("=" * 60)
print(f"\nQuery: {query}\n")

initial_state = {
    "original_query": query,
    "current_query": query,
    "sub_questions": [],
    "documents": [],
    "generation": "",
    "filters": {},
    "retry_count": 0,
    "revision_count": 0,
    "grade_status": "",
    "hallucination_status": "",
    "critique": ""
}

# Run with simpler query to speed up testing
try:
    final_state = None
    for i, state in enumerate(app.stream(initial_state)):
        print(f"\n--- Step {i+1} ---")
        final_state = state
        if i > 10:  # Safety limit
            print("\nReached step limit, stopping...")
            break
    
    # Get final answer
    if final_state:
        for node_name, node_state in final_state.items():
            if "generation" in node_state and node_state["generation"]:
                print("\n" + "=" * 60)
                print("FINAL ANSWER:")
                print("=" * 60)
                print(node_state["generation"])
                print("=" * 60)
                break
except KeyboardInterrupt:
    print("\n\nTest interrupted by user")
except Exception as e:
    print(f"\n\nError during test: {e}")
    import traceback
    traceback.print_exc()
