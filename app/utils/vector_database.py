import os
import numpy as np
import faiss
import pickle
import json

class VectorDatabase:
    def __init__(self, dimension=None):
        """
        Initialize a vector database using FAISS
        
        Args:
            dimension: Dimension of the embeddings to store (required for initialization)
        """
        self.index = None
        self.dimension = dimension
        self.metadata = []
        self.product_data = None
    
    def create_index(self, embeddings=None):
        """Create a new index with the given dimension"""
        if embeddings is not None:
            embeddings = np.asarray(embeddings)
            self.dimension = embeddings.shape[1]
        
        if self.dimension is None:
            raise ValueError("Dimension must be provided either during initialization or when creating index")
        
        # Create a flat L2 index (exact search)
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # If embeddings are provided, add them to the index
        if embeddings is not None and len(embeddings) > 0:
            self.add_embeddings(embeddings)
            
        return self
    
    def add_embeddings(self, embeddings, metadata_list=None):
        """
        Add embeddings to the index
        
        Args:
            embeddings: Numpy array of embeddings to add
            metadata_list: List of metadata dictionaries for each embedding
        """
        if self.index is None:
            self.create_index(embeddings)
            return self
        
        # Ensure embeddings are in the correct format (float32)
        embeddings = np.asarray(embeddings).astype(np.float32)
        
        # Add embeddings to the FAISS index
        self.index.add(embeddings)
        
        # Store metadata if provided
        if metadata_list:
            self.metadata.extend(metadata_list)
        else:
            # Create empty metadata if not provided
            self.metadata.extend([{} for _ in range(len(embeddings))])
            
        return self
    
    def search(self, query_embedding, k=5):
        """
        Search for similar embeddings in the index
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of results with metadata
        """
        if self.index is None:
            raise ValueError("Index has not been created yet")
        
        # Ensure query_embedding is in the correct format
        query_embedding = np.asarray(query_embedding).astype(np.float32)
        
        # Reshape if necessary
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return results with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):  # Ensure valid index
                distance = float(distances[0][i]) if i < len(distances[0]) else float('inf')
                result = {
                    "distance": distance,
                    "index": int(idx),
                    "metadata": self.metadata[idx]
                }
                
                # Add full product data if available
                if self.product_data is not None and 'id' in self.metadata[idx]:
                    product = self.product_data[self.product_data['id'] == self.metadata[idx]['id']]
                    if not product.empty:
                        result["product"] = product.iloc[0].to_dict()
                        
                results.append(result)
                
        return results
    
    def load_product_data(self, products_df):
        """Load product data for enriching search results"""
        self.product_data = products_df
        return self
    
    def save(self, directory, prefix="vector_db"):
        """Save the vector database to disk"""
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, f"{prefix}_index.faiss"))
        
        # Save metadata
        with open(os.path.join(directory, f"{prefix}_metadata.pkl"), 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save config
        with open(os.path.join(directory, f"{prefix}_config.json"), 'w') as f:
            json.dump({"dimension": self.dimension}, f)
        
        return self
    
    @classmethod
    def load(cls, directory, prefix="vector_db"):
        """Load a vector database from disk"""
        # Load config
        with open(os.path.join(directory, f"{prefix}_config.json"), 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(dimension=config["dimension"])
        
        # Load FAISS index
        instance.index = faiss.read_index(os.path.join(directory, f"{prefix}_index.faiss"))
        
        # Load metadata
        with open(os.path.join(directory, f"{prefix}_metadata.pkl"), 'rb') as f:
            instance.metadata = pickle.load(f)
        
        return instance 