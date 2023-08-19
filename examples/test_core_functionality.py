from littlevectordb.core import LittleVectorDB


def test_littlevectordb():

    # Create a new database instance
    db = LittleVectorDB()

    # Insert some vectors
    db.insert([0, 0, 0], vector_id=100)  # Insert a vector with a specific ID
    assert db.delete(100) == True  # Ensure the vector is deleted successfully
    assert db.delete(100) == False  # Ensure the vector can't be deleted again
    # Test batch insertion
    
    batch_vectors = [[4, 5, 6], [6, 5, 4], [4, 4, 4]]
    ids = db.batch_insert(batch_vectors)
    assert len(ids) == 3  # Ensure all vectors were inserted

    # Test batch querying
    query_vectors = [[4, 5, 6], [1, 2, 3]]
    results = db.batch_query(query_vectors, top_k=2)
    assert len(results) == 2  # Ensure results are returned for each query vector

    # Save the database to a file
    db.save("test_db.pkl")

    # Load the database from the saved file
    loaded_db = LittleVectorDB.load("test_db.pkl")

    # Query the loaded database
    results = loaded_db.query([1, 2, 3], top_k=2)
    print(results)

    # Clean up (optional: remove the saved file after testing)
    import os
    os.remove("test_db.pkl")

if __name__ == "__main__":
    test_littlevectordb()
