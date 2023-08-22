from littlevectordb.core import LittleVectorDB
import time

def test_littlevectordb():
    # Test for cosine similarity with Annoy
    db_cosine = LittleVectorDB(distance_metric='cosine')
    _run_tests(db_cosine, "Cosine Similarity with Annoy")

    # Test for Euclidean distance with Annoy
    db_euclidean = LittleVectorDB(distance_metric='euclidean')
    _run_tests(db_euclidean, "Euclidean Distance with Annoy")


def _run_tests(db, metric_name):
    print("=" * 40)
    print(f"Running tests for {metric_name}...")
    print("-" * 40)

    # Insert some vectors
    db.insert([0, 0, 0], vector_id=100)  # Insert a vector with a specific ID
    assert db.delete(100) == True  # Ensure the vector is deleted successfully
    assert db.delete(100) == False  # Ensure the vector can't be deleted again
    
    # Test batch insertion
    batch_vectors = [[4, 5, 6], [6, 5, 4], [4, 4, 4]]
    ids = db.batch_insert(batch_vectors)
    assert len(ids) == 3  # Ensure all vectors were inserted

    # Build Annoy index
    db.build_index()

    # Test batch querying with Annoy index
    start_time = time.time()
    query_vectors = [[4, 5, 6], [1, 2, 3]]
    results_indexed = db.batch_query(query_vectors, top_k=2)
    indexed_time = time.time() - start_time
    print(f"Time taken for indexed query: {indexed_time:.4f} seconds")
    assert len(results_indexed) == 2  # Ensure results are returned for each query vector

    # Test batch querying without Annoy index
    start_time = time.time()
    results_bruteforce = db.batch_query(query_vectors, top_k=2, use_index=False)
    bruteforce_time = time.time() - start_time
    print(f"Time taken for brute-force query: {bruteforce_time:.4f} seconds")
    assert len(results_bruteforce) == 2  # Ensure results are returned for each query vector

    # Assert that both methods return similar results
    for r1, r2 in zip(results_indexed, results_bruteforce):
        assert [i[0] for i in r1] == [i[0] for i in r2]

    # Save the database to a file
    db.save("test_db.pkl")

    # Load the database from the saved file
    loaded_db = LittleVectorDB.load("test_db.pkl")

    # Query the loaded database with Annoy index
    results = loaded_db.query([1, 2, 3], top_k=2)
    # print(results) # Diagnostic Print

    # Clean up (optional: remove the saved file after testing)
    import os
    os.remove("test_db.pkl")

    print(f"Tests for {metric_name} completed successfully!")

if __name__ == "__main__":
    test_littlevectordb()
