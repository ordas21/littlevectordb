import spacy
import time
from littlevectordb.core import LittleVectorDB

def word_embeddings_example():
    # Load the spaCy model for word embeddings
    nlp = spacy.load("en_core_web_md")

    # Words to insert into the database
    words = ["king", "queen", "man", "woman", "apple", "orange", "fruit", "computer", "laptop"]

    # Query words for testing
    query_words = ["monarch", "lady", "pc"]

    # Test for each metric
    for metric in ["cosine", "euclidean"]:
        print("=" * 40)
        print(f"Testing for {metric.upper()} metric:")
        print("-" * 40)

        # Create a new database instance with the specified metric
        db = LittleVectorDB(distance_metric=metric)

        # Insert word embeddings into the database
        for word in words:
            vector = nlp(word).vector
            db.insert(vector, vector_id=word)

        # Build the Annoy index
        db.build_index()

        # Query the database using the Annoy index
        print("Using Annoy Index:")
        start_time = time.time()
        for word in query_words:
            vector = nlp(word).vector
            results = db.query(vector, top_k=3)
            print(f"Words similar to '{word}': {results}")
        indexed_time = time.time() - start_time
        print(f"Time taken with Annoy index: {indexed_time:.4f} seconds")
        print()

        # Query the database using brute-force method
        print("Using Brute-Force:")
        start_time = time.time()
        for word in query_words:
            vector = nlp(word).vector
            results = db.query(vector, top_k=3, use_index=False)
            print(f"Words similar to '{word}': {results}")
        bruteforce_time = time.time() - start_time
        print(f"Time taken with brute-force: {bruteforce_time:.4f} seconds")
        print()
        

if __name__ == "__main__":
    word_embeddings_example()
