import numpy as np
import pickle
from scipy.spatial import KDTree
from annoy import AnnoyIndex
import os


class LittleVectorDB:
    """
    LittleVectorDB: A minimalistic vector database.
    
    Features:
    - Lightweight: Optimized for minimal memory and CPU usage.
    - Transparent: Clear and understandable operations.
    - Embedding Agnostic: Accepts vectors of any dimension.
    - Zero Configuration: Sensible defaults for all operations.
    """

    def __init__(self, distance_metric='cosine'):
        """
        Initialize an empty database.
        """
        self.vectors = []
        self.ids = []
        self.id_to_index = {}  # New dictionary to map vector IDs to their indices
        self.kdtree = None  # KD-tree for advanced indexing
        self.distance_metric = distance_metric
        self.annoy_index = None
        self.annoy_metric = 'angular' if distance_metric == 'cosine' else 'euclidean'

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
    
    def _euclidean_distance(self, v1, v2):
        """
        Compute the Euclidean distance between two vectors.
        
        Parameters:
        - v1, v2 (numpy arrays): The vectors to be compared.
        
        Returns:
        - float: The Euclidean distance.
        """
        return np.linalg.norm(v1 - v2)

    def build_index(self, n_trees=10):
        """
        Build an Annoy index for faster similarity searches.
        
        Parameters:
        - n_trees (int): Number of trees for the Annoy index. More trees give higher precision.
        """
        dimension = len(self.vectors[0]) if self.vectors else 0
        self.annoy_index = AnnoyIndex(dimension, self.annoy_metric)
        
        for i, vector in enumerate(self.vectors):
            self.annoy_index.add_item(i, vector)
        
        self.annoy_index.build(n_trees)
    
    def _normalize_vector(self, vector):
        """
        Normalize a vector to have a magnitude (L2 norm) of 1.
        
        Parameters:
        - vector (numpy array): The vector to be normalized.
        
        Returns:
        - numpy array: The normalized vector.
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def insert(self, vector, vector_id=None):
        """
        Insert a vector into the database.
        
        Parameters:
        - vector (list or numpy array): The vector to be inserted.
        - vector_id (optional): A unique identifier for the vector. If not provided, an auto-incremented ID is used.
        
        Returns:
        - int: The ID associated with the inserted vector.
        """
        normalized_vector = self._normalize_vector(np.array(vector))
        if vector_id is None:
            vector_id = len(self.vectors)
        self.vectors.append(normalized_vector)
        self.ids.append(vector_id)
        self.id_to_index[vector_id] = len(self.vectors) - 1  # Update the mapping
        return vector_id

    def query(self, vector, top_k=5, use_index=True):
        """
        Find the top_k most similar vectors in the database.

        Parameters:
        - vector (list or numpy array): The query vector.
        - top_k (int): Number of top similar vectors to retrieve.
        - use_index (bool): Whether to use the Annoy index for the search.

        Returns:
        - list: A list of tuples, where each tuple is (vector_id, similarity_score or distance).
        """
        if use_index and self.annoy_index:
            indices, distances = self.annoy_index.get_nns_by_vector(vector, top_k, include_distances=True)
            # print("Indices from Annoy:", indices)  # Diagnostic print
            # print("Length of self.ids:", len(self.ids))  # Diagnostic print
            if self.annoy_metric == 'angular':
                # Convert angular distance to cosine similarity
                similarities = [1 - (d / 2) for d in distances]
                return [(self.ids[indices[i]], similarities[i]) for i in range(len(indices))]
            else:
                return [(self.ids[i], d) for i, d in zip(indices, distances)]
        else:
            vector = np.array(vector)
            if self.distance_metric == 'cosine':
                scores = [self._cosine_similarity(vector, v) for v in self.vectors]
                sorted_indices = np.argsort(scores)[::-1]
                return [(self.ids[i], scores[i]) for i in sorted_indices[:top_k]]
            elif self.distance_metric == 'euclidean':
                distances = [self._euclidean_distance(vector, v) for v in self.vectors]
                sorted_indices = np.argsort(distances)
                return [(self.ids[i], distances[i]) for i in sorted_indices[:top_k]]
            else:
                raise ValueError(f"Unsupported metric: {self.distance_metric}")
        
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
        if self.annoy_index:
            annoy_filename = filename + ".annoy"
            self.annoy_index.save(annoy_filename)
            self.annoy_index = None  # Temporarily set to None for pickling

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

        if annoy_filename:
            self.build_index()  # Rebuild the Annoy index after pickling

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
            db = pickle.load(file)

        annoy_filename = filename + ".annoy"
        if os.path.exists(annoy_filename):
            db.annoy_index = AnnoyIndex(len(db.vectors[0]) if db.vectors else 0, db.annoy_metric)
            db.annoy_index.load(annoy_filename)

        return db

    def batch_insert(self, vectors, vector_ids=None):
            """
            Insert multiple vectors into the database at once.

            Parameters:
            - vectors (list of lists or numpy array): The vectors to be inserted.
            - vector_ids (list, optional): A list of unique identifiers for the vectors. 
                                        If not provided, auto-incremented IDs are used.

            Returns:
            - list: The IDs associated with the inserted vectors.
            """
            if vector_ids is None:
                vector_ids = list(range(len(self.vectors), len(self.vectors) + len(vectors)))

            self.vectors.extend(vectors)
            self.ids.extend(vector_ids)

            return vector_ids

    def batch_query(self, vectors, top_k=5, use_index=True):
        """
        Find the top_k most similar vectors in the database for each query vector.

        Parameters:
        - vectors (list of lists or numpy array): The query vectors.
        - top_k (int): Number of top similar vectors to retrieve for each query.
        - use_index (bool): Whether to use the Annoy index for the search.

        Returns:
        - list of lists: Each inner list contains tuples for a query vector, 
                        where each tuple is (vector_id, similarity_score).
        """
        return [self.query(vector, top_k, use_index=use_index) for vector in vectors]