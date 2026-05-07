import sys
from build_index import build_all_indexes, index_exists
from query_engine import run_query

def main():
    # Parse flags
    do_evaluate = "--evaluate" in sys.argv
    do_rebuild = "--rebuild" in sys.argv
    
    # Handle rebuild if requested or if index doesn't exist
    if do_rebuild or not index_exists("code_functions"):
        if do_rebuild:
            print("Rebuilding index...")
        else:
            print("No index found. Building automatically...")
        build_all_indexes("test_repo", "payment_flow_fixed.png", repo_hash="test_repo_v1")
    else:
        print("Using existing index.")
    
    # Run evaluation or normal query
    if do_evaluate:
        print("\nRunning evaluation mode...")
        try:
            from evaluate import run_evaluation
            run_evaluation()
        except ImportError as e:
            print(f"Error: Could not import evaluation module - {e}")
            sys.exit(1)
    else:
        if index_exists("code_functions"):
            run_query()
        else:
            print("Failed to build index. Exiting.")

if __name__ == "__main__":
    main()