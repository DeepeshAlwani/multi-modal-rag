import sys
from build_index import build_all_indexes, index_exists
from query_engine import run_query

def main():
    if "--rebuild" in sys.argv:
        print("Rebuilding index...")
        build_all_indexes("test_repo", "payment_flow_fixed.png")
    elif not index_exists("code_functions"):
        print("No index found. Building automatically...")
        build_all_indexes("test_repo", "payment_flow_fixed.png")
    else:
        print("Using existing index.")
    
    # Only run query if index exists (or after building)
    if index_exists("code_functions"):
        run_query()
    else:
        print("Failed to build index. Exiting.")

if __name__ == "__main__":
    main()