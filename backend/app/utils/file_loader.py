"""
Common utilities for RAG search methods
"""

import os
import glob


def read_wiki_docs(path="./app/data/**/*.txt.clean"):
    """Read all documents from data directory"""
    docs = []
    doc_paths = []

    files = glob.glob(path, recursive=True)
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:  # Only add non-empty files
                    docs.append(content)
                    doc_paths.append(file_path)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

    return docs, doc_paths


def get_doc_info(path):
    """Get document information for display"""
    docs, paths = read_wiki_docs(path)

    print(f"📚 Loaded {len(docs)} documents")
    print("\nDocuments:")
    for i, (doc, path) in enumerate(zip(docs, paths)):
        # Get relative path for cleaner display
        rel_path = path.replace("./app", "./")
        print(f"{i + 1}. [{rel_path}] {doc[:80]}{'...' if len(doc) > 80 else ''}")

    return docs, paths


if __name__ == "__main__":
    docs, paths = get_doc_info("../data/**/*.txt.clean")
    print(paths)
