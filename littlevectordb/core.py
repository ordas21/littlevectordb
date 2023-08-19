import numpy as np
import pickle

class LittleVectorDB:
    """
    LittleVectorDB: A minimalistic vector database.
    
    Features:
    - Lightweight: Optimized for minimal memory and CPU usage.
    - Transparent: Clear and understandable operations.
    - Embedding Agnostic: Accepts vectors of any dimension.
    - Zero Configuration: Sensible defaults for all operations.
    """

    def __init__(self):
        """
        Initialize an empty database.
        """
        self.vectors = []
        self.ids = []
        self.id_to_index = {}  # New dictionary to map vector IDs to their indices
        
    def insert(self, vector, vector_id=None):
        """
        Insert a vector into the database.
        
        Parameters:
        - vector (list or numpy array): The vector to be inserted.
        - vector_id (optional): A unique identifier for the vector. If not provided, an auto-incremented ID is used.
        
        Returns:
        - int: The ID associated with the inserted vector.
        """
        if vector_id is None:
            vector_id = len(self.vectors)
        self.vectors.append(np.array(vector))
        self.ids.append(vector_id)

        self.id_to_index[vector_id] = len(self.vectors) - 1  # Update the mapping

        return vector_id

    def query(self, vector, top_k=5):
        """
        Find the top_k most similar vectors in the database using cosine similarity.
        
        Parameters:
        - vector (list or numpy array): The query vector.
        - top_k (int): Number of top similar vectors to retrieve.
        
        Returns:
        - list of tuples: Each tuple contains (vector_id, similarity_score).
        """
        vector = np.array(vector)
        similarities = [self._cosine_similarity(vector, v) for v in self.vectors]
        sorted_indices = np.argsort(similarities)[::-1]
        return [(self.ids[i], similarities[i]) for i in sorted_indices[:top_k]]
    
    def delete(self, vector_id):
        """
        Delete a vector from the database based on its unique ID.

        Parameters:
        - vector_id (int): The ID of the vector to be deleted.

        Returns:
        - bool: True if the vector was successfully deleted, False otherwise.
        """
        if vector_id in self.id_to_index:
            index = self.id_to_index[vector_id]
            del self.vectors[index]
            del self.ids[index]
            del self.id_to_index[vector_id]
            
            # Update indices in the id_to_index mapping
            for i, vid in enumerate(self.ids[index:]):
                self.id_to_index[vid] = index + i

            return True
        return False

    def _cosine_similarity(self, v1, v2):
        """
        Compute the cosine similarity between two vectors.
        
        Parameters:
        - v1, v2 (numpy arrays): The vectors to be compared.
        
        Returns:
        - float: The cosine similarity score.
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)
        
    def save(self, filename):
            """
            Save the current state of the database to a file.
            
            Parameters:
            - filename (str): The path to the file where the database should be saved.
            """
            with open(filename, 'wb') as file:
                pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        """
        Load the database from a saved file.
        
        Parameters:
        - filename (str): The path to the file from which the database should be loaded.
        
        Returns:
        - LittleVectorDB: An instance of the database loaded from the file.
        """
        with open(filename, 'rb') as file:
            return pickle.load(file)
        

