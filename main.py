import sys
from parse_functions import parse_documents
from build_index import build_index, index_exists
from query_engine import run_query

def main() -> None:
    if "--rebuild" in sys.argv:
        print("Rebuilding index...")
        documents = parse_documents("test_repo")
        build_index(documents)
    elif not index_exists():
        print("No index found. Building automatically...")
        documents = parse_documents("test_repo")
        build_index(documents)
    else:
        print("Using existing index.")
run_query()

if __name__ == "__main__":
    main()