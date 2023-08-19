from littlevectordb.core import LittleVectorDB


def test_littlevectordb():

    # Create a new database instance
    db = LittleVectorDB()

    # Insert some vectors
    db.insert([0, 0, 0], vector_id=100)  # Insert a vector with a specific ID
    assert db.delete(100) == True  # Ensure the vector is deleted successfully
    assert db.delete(100) == False  # Ensure the vector can't be deleted again

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
