import spacy
from littlevectordb.core import LittleVectorDB

def word_embeddings_example():
    # Load spaCy's medium-sized English model with word vectors
    nlp = spacy.load("en_core_web_md")

    # Create an instance of LittleVectorDB
    db = LittleVectorDB()

    # List of words to insert into the database
    words = ["king", "queen", "man", "woman", "child", "royalty", "monarch", "prince", "princess"]

    # Insert word embeddings into the database
    for word in words:
        vector = nlp(word).vector
        db.insert(vector, vector_id=word)

    # Query the database with the embedding of the word "king"
    query_vector = nlp("king").vector
    similar_words = db.query(query_vector, top_k=5)

    # Display the results
    print("Words similar to 'king':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")

if __name__ == "__main__":
    word_embeddings_example()
