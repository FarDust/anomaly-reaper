from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Callable,
)
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    PrivateAttr,
    model_validator,
)
from hashlib import md5
import array
import struct



class ImagesVectorStore(VectorStore, BaseModel):
    """A vector store for image embeddings using the LangChain VectorStore interface.
    
    This implementation works with image embeddings stored in various formats (CSV files, 
    dataframes, or numpy arrays) and provides similarity search functionality to find 
    images with similar embeddings.
    
    The store maintains a mapping between image embeddings and their file paths, allowing
    for efficient retrieval and search operations. It supports adding, deleting, and
    searching for images based on embeddings or paths.
    
    Parameters
    ----------
    embeddings_df : Optional[list[list[float]]], default=None
        List of image embedding vectors, where each vector is represented as a list of floats.
    image_paths_list : Optional[list[str]], default=None
        List of image paths or URIs corresponding to the embeddings.
    embedding_function : Optional[Embeddings], default=None
        Function to generate embeddings for new images.
        
    Attributes
    ----------
    _embeddings_to_paths : dict[tuple[float, ...], str]
        Internal mapping from embedding tuples (hashable) to image paths.
    _embeddings_to_metadata : dict[tuple[float, ...], dict]
        Internal mapping from embedding tuples to metadata dictionaries.
        
    Notes
    -----
    - Both embeddings_df and image_paths_list must have the same length.
    - For efficient searching, the embeddings are converted to tuples for hashability.
    - Supports multiple search methods including regular similarity search and 
      maximal marginal relevance search for diversity.
    """
    
    embeddings_df: list[list[float]] = Field() # DataFrame containing image embeddings
    image_paths_list: list[str] = Field() # List of image paths or gspaths corresponding to the embeddings
    embedding_function: Embeddings = Field() # For generating embeddings for new images

    # Use tuples of floats (hashable) for the embedding keys
    _embeddings_to_paths: dict[tuple[float, ...], str] = PrivateAttr(default_factory=dict)
    _embeddings_to_metadata: dict[tuple[float, ...], dict] = PrivateAttr(default_factory=dict)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @model_validator(mode='after')
    def validate_embeddings_and_paths(self) -> 'ImagesVectorStore':
        """
        Validates that embeddings and image paths are properly defined and matched.
        Also initializes the internal embedding mapping dictionary.
        """
        # Ensure embeddings_df and image_paths_list are both provided
        if self.embeddings_df is None or self.image_paths_list is None:
            raise ValueError("Both embeddings_df and image_paths_list must be provided")
            
        # Validate dimensions
        if len(self.embeddings_df) != len(self.image_paths_list):
            raise ValueError(
                f"Number of embeddings ({len(self.embeddings_df)}) does not match number of image paths ({len(self.image_paths_list)})"
            )
            
        # Initialize the embeddings mapping dictionary using embeddings as keys
        for i, embedding in enumerate(self.embeddings_df):
            # Convert list[float] to tuple[float, ...] for hashability
            embedding_key = tuple(embedding)
            path = self.image_paths_list[i]
            self._embeddings_to_paths[embedding_key] = path
            
        return self

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add image paths or base64 strings to the vector store and generate embeddings for them.
        
        Args:
            texts: List of image paths (local or gs://) or base64-encoded image strings
            metadatas: Optional metadata for each image
            ids: Optional IDs for each image
            **kwargs: Additional arguments
            
        Returns:
            List of IDs for the added images
        """
        if self.embedding_function is None:
            raise ValueError("embedding_function must be provided to add new images")
        
        # Convert to list if not already
        texts_list = list(texts)
        
        # Ensure metadatas is properly handled
        if metadatas is None:
            metadatas = [{} for _ in texts_list]
        elif len(metadatas) != len(texts_list):
            raise ValueError(f"Got {len(metadatas)} metadatas for {len(texts_list)} texts")
        
        # Generate embeddings for all images
        new_embeddings = self.embedding_function.embed_documents(texts_list)
        
        # Add to our data structures
        if self.embeddings_df is None:
            self.embeddings_df = new_embeddings
            self.image_paths_list = texts_list
        else:
            self.embeddings_df.extend(new_embeddings)
            self.image_paths_list.extend(texts_list)
        
        # Update the mapping and metadata
        for i, path in enumerate(texts_list):
            # Convert list[float] to tuple[float, ...] for hashability
            embedding_key = tuple(new_embeddings[i])
            self._embeddings_to_paths[embedding_key] = path
            # Store the metadata separately - this was the issue
            if i < len(metadatas):
                self._embeddings_to_metadata[embedding_key] = metadatas[i]
        
        # Generate or use provided IDs
        if ids is None:
            ids = [str(i + len(self.embeddings_df) - len(texts_list)) for i in range(len(texts_list))]
        
        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        Delete images by vector ID or other criteria.
        
        Args:
            ids: List of IDs to delete. These can be MD5 hashes of embedding tuples
                 or UUIDs. If None, delete all. Default is None.
            **kwargs: Other arguments for filtering, can include:
                path_pattern: Regex pattern to match against image paths
            
        Returns:
            True if successful, False otherwise
        """
        
        # Helper function to hash embedding vectors efficiently
        def hash_embedding(embedding):
            # Convert embedding to bytes directly without string conversion
            # This is much more efficient for large vectors
            # Use a memory-efficient approach with array and struct
            try:
                # Convert floats to bytes directly
                float_array = array.array('d', embedding)
                bytes_data = float_array.tobytes()
                return md5(bytes_data).hexdigest()
            except:
                # Fallback if the above doesn't work
                return md5(struct.pack('<%sd' % len(embedding), *embedding)).hexdigest()
        
        # Clear everything if no filters provided
        if ids is None and not kwargs:
            # Clear everything
            self.embeddings_df = []
            self.image_paths_list = []
            self._embeddings_to_paths.clear()
            self._embeddings_to_metadata.clear()
            return True
        
        # Handle deletion by IDs (UUIDs or MD5 hashes)
        if ids is not None:
            indices_to_delete = []
            
            # Process each ID (treating as MD5 hash of embedding)
            for id_str in ids:
                # Try to match as MD5 hash of embedding
                for i, embedding in enumerate(self.embeddings_df):
                    # Hash the embedding efficiently
                    current_hash = hash_embedding(embedding)
                    
                    if current_hash == id_str:
                        indices_to_delete.append(i)
                        # Remove from mappings
                        embedding_key = tuple(embedding)
                        if embedding_key in self._embeddings_to_paths:
                            del self._embeddings_to_paths[embedding_key]
                        if embedding_key in self._embeddings_to_metadata:
                            del self._embeddings_to_metadata[embedding_key]
            
            # Create new lists excluding deleted items
            if indices_to_delete:
                self.embeddings_df = [
                    emb for i, emb in enumerate(self.embeddings_df) 
                    if i not in indices_to_delete
                ]
                self.image_paths_list = [
                    path for i, path in enumerate(self.image_paths_list) 
                    if i not in indices_to_delete
                ]
                return True
            
            return False
        
        # Handle path pattern filtering
        path_pattern = kwargs.get("path_pattern")
        if path_pattern:
            import re
            pattern = re.compile(path_pattern)
            
            indices_to_delete = []
            for i, path in enumerate(self.image_paths_list):
                if pattern.search(path):
                    indices_to_delete.append(i)
                    embedding_key = tuple(self.embeddings_df[i])
                    if embedding_key in self._embeddings_to_paths:
                        del self._embeddings_to_paths[embedding_key]
                    if embedding_key in self._embeddings_to_metadata:
                        del self._embeddings_to_metadata[embedding_key]
            
            # Create new lists excluding deleted items
            self.embeddings_df = [
                emb for i, emb in enumerate(self.embeddings_df) if i not in indices_to_delete
            ]
            self.image_paths_list = [
                path for i, path in enumerate(self.image_paths_list) if i not in indices_to_delete
            ]
            
            return True
        
        return False

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar images based on a query image path or URI.
        
        This method finds the most similar images to a query image by converting
        the query to an embedding vector and performing a similarity search.
        
        Parameters
        ----------
        query : str
            Path, URL, or URI to the query image. If this image already exists in the
            store, its existing embedding will be used. Otherwise, the embedding_function
            will be used to generate a new embedding.
        k : int, default=4
            Number of similar images to return.
        **kwargs : Any
            Additional arguments for the search, such as:
            - metadata_callback: A function that takes (path, index) and returns
              additional metadata to include in the Document.
            
        Returns
        -------
        List[Document]
            A list of Document objects, each containing:
            - page_content: The path to a similar image
            - metadata: A dictionary with at least a 'source' key pointing to the image
            
        Examples
        --------
        >>> from anomaly_reaper.infrastructure.vector_database.images_store import ImagesVectorStore
        >>> 
        >>> # Assuming 'store' is an already initialized ImagesVectorStore
        >>> results = store.similarity_search("data/query_image.png", k=3)
        >>> 
        >>> # Print the results
        >>> for doc in results:
        >>>     print(f"Similar image: {doc.page_content}")
        >>>     print(f"Source: {doc.metadata['source']}")
        >>>     print("---")
        
        Notes
        -----
        - If the query image is not found in the store and no embedding_function is
          provided, a ValueError will be raised.
        - This method returns the Documents without scores. For results with similarity
          scores, use similarity_search_with_score() instead.
        """
        # Get search results with scores
        results = self.similarity_search_with_score(query, k, **kwargs)
        
        # Return just the documents without scores
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar images with similarity scores based on a query image.
        
        This method finds the most similar images to a query image by converting
        the query to an embedding vector and performing a similarity search with scores.
        
        Parameters
        ----------
        query : str
            Path, URL, or URI to the query image. If this image already exists in the
            store, its existing embedding will be used. Otherwise, the embedding_function
            will be used to generate a new embedding.
        k : int, default=4
            Number of similar images to return.
        **kwargs : Any
            Additional arguments for the search, such as:
            - metadata_callback: A function that takes (path, index) and returns
              additional metadata to include in the Document.
            
        Returns
        -------
        List[Tuple[Document, float]]
            A list of tuples, each containing:
            - Document: Contains the path and metadata of a similar image
            - float: Similarity score (higher means more similar)
            
        Examples
        --------
        >>> from anomaly_reaper.infrastructure.vector_database.images_store import ImagesVectorStore
        >>> 
        >>> # Assuming 'store' is an already initialized ImagesVectorStore
        >>> results = store.similarity_search_with_score("data/query_image.png", k=3)
        >>> 
        >>> # Print the results with scores
        >>> for doc, score in results:
        >>>     print(f"Similar image: {doc.page_content}")
        >>>     print(f"Similarity score: {score:.4f}")
        >>>     print("---")
        
        Notes
        -----
        - If the query image is not found in the store and no embedding_function is
          provided, a ValueError will be raised.
        - The similarity scores are cosine similarity values, ranging from -1 to 1,
          where higher values indicate greater similarity.
        """
        # Convert query to embedding if it's an image path or base64 string
        query_embedding = None
        
        # Case 1: Query is a path that exists in our mapping
        if query in self.image_paths_list:
            # Get the index of the path
            idx = self.image_paths_list.index(query)
            # Get the embedding
            query_embedding = self.embeddings_df[idx]
        
        # Case 2: Query is a path to an image we haven't seen, but we have an embedding function
        elif self.embedding_function is not None:
            query_embedding = self.embedding_function.embed_query(query)
        
        # If we couldn't get an embedding, raise an error
        if query_embedding is None:
            raise ValueError(
                "Could not generate embedding for query. "
                "Either the query must be a known image path or "
                "embedding_function must be provided."
            )
        
        # Now we can do the vector-based search
        return self.similarity_search_by_vector_with_score(query_embedding, k, **kwargs)
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar images using a vector embedding directly.
        
        This method finds the most similar images to a query embedding vector
        without requiring a query image path. Useful when you already have
        an embedding vector from another source.
        
        Parameters
        ----------
        embedding : List[float]
            The embedding vector to search with. Must have the same dimensions
            as the embeddings stored in the vector store.
        k : int, default=4
            Number of similar images to return.
        **kwargs : Any
            Additional arguments for the search, such as:
            - metadata_callback: A function that takes (path, index) and returns
              additional metadata to include in the Document.
            
        Returns
        -------
        List[Document]
            A list of Document objects, each containing:
            - page_content: The path to a similar image
            - metadata: A dictionary with at least a 'source' key pointing to the image
            
        Examples
        --------
        >>> from anomaly_reaper.infrastructure.vector_database.images_store import ImagesVectorStore
        >>> import numpy as np
        >>> 
        >>> # Assuming 'store' is an already initialized ImagesVectorStore
        >>> # And we have a query embedding from somewhere
        >>> query_embedding = np.random.rand(512).tolist()  # Example with random vector
        >>> 
        >>> results = store.similarity_search_by_vector(query_embedding, k=3)
        >>> for doc in results:
        >>>     print(f"Similar image: {doc.page_content}")
        
        Notes
        -----
        - The query embedding must have the same dimension as the embeddings in the store.
        - This method returns the Documents without scores. For results with similarity
          scores, use similarity_search_by_vector_with_score() instead.
        """
        results = self.similarity_search_by_vector_with_score(embedding, k, **kwargs)
        return [doc for doc, _ in results]
    
    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar images with similarity scores using a vector embedding directly.
        
        This method finds the most similar images to a query embedding vector
        and returns both the documents and their similarity scores.
        
        Parameters
        ----------
        embedding : List[float]
            The embedding vector to search with. Must have the same dimensions
            as the embeddings stored in the vector store.
        k : int, default=4
            Number of similar images to return.
        **kwargs : Any
            Additional arguments for the search, such as:
            - metadata_callback: A function that takes (path, index) and returns
              additional metadata to include in the Document.
            
        Returns
        -------
        List[Tuple[Document, float]]
            A list of tuples, each containing:
            - Document: Contains the path and metadata of a similar image
            - float: Similarity score (higher means more similar)
            
        Examples
        --------
        >>> from anomaly_reaper.infrastructure.vector_database.images_store import ImagesVectorStore
        >>> import numpy as np
        >>> 
        >>> # Assuming 'store' is an already initialized ImagesVectorStore
        >>> # And we have a query embedding from somewhere
        >>> query_embedding = np.random.rand(512).tolist()  # Example with random vector
        >>> 
        >>> results = store.similarity_search_by_vector_with_score(query_embedding, k=3)
        >>> for doc, score in results:
        >>>     print(f"Similar image: {doc.page_content}")
        >>>     print(f"Similarity score: {score:.4f}")
        >>>     print("---")
        
        Notes
        -----
        - The query embedding must have the same dimension as the embeddings in the store.
        - The similarity scores are cosine similarity values, ranging from -1 to 1,
          where higher values indicate greater similarity.
        - This method is the backbone of most other search methods in the class.
        """
        
        # Convert our embedding lists to numpy arrays for similarity calculation
        query_embedding_array = np.array(embedding).reshape(1, -1)
        embeddings_array = np.array(self.embeddings_df)
        
        # Calculate similarity
        similarities = cosine_similarity(query_embedding_array, embeddings_array)[0]
        
        # Get indices of top-k most similar embeddings
        if k > len(similarities):
            k = len(similarities)
        
        # Process for retrieval
        top_indices = np.argsort(similarities)[-k:][::-1]  # Get highest similarity indices
        
        # Get metadata_callback if provided
        metadata_callback = kwargs.get("metadata_callback")
        
        results = []
        for idx in top_indices:
            # Get embedding and path
            embedding_vector = self.embeddings_df[idx]
            embedding_key = tuple(embedding_vector)
            path = self.image_paths_list[idx]
            
            # Start with stored metadata if available
            if embedding_key in self._embeddings_to_metadata:
                metadata = self._embeddings_to_metadata[embedding_key].copy()
            else:
                metadata = {}
            
            # Only include source if not already in metadata
            if "source" not in metadata:
                metadata["source"] = path
            
            # Use callback for additional metadata if provided
            if metadata_callback and callable(metadata_callback):
                try:
                    user_metadata = metadata_callback(path, idx)
                    if user_metadata and isinstance(user_metadata, dict):
                        metadata.update(user_metadata)
                except Exception as e:
                    print(f"Error in metadata callback: {e}")
            
            # Create document
            doc = Document(page_content=path, metadata=metadata)
            score = float(similarities[idx])
            
            results.append((doc, score))
        
        return results
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for diverse but relevant images based on a query image.
        
        This method uses maximal marginal relevance (MMR) to find images that are both
        relevant to the query and diverse among themselves. It balances similarity to the
        query with diversity in the result set.
        
        Parameters
        ----------
        query : str
            Path, URL, or URI to the query image. If this image already exists in the
            store, its existing embedding will be used. Otherwise, the embedding_function
            will be used to generate a new embedding.
        k : int, default=4
            Number of diverse documents to return.
        fetch_k : int, default=20
            Number of documents to initially fetch before applying diversity filtering.
            Should be greater than or equal to k.
        lambda_mult : float, default=0.5
            Balance factor between relevance and diversity. Ranges from 0 to 1:
            - 0: Focus purely on diversity (may return less relevant results)
            - 1: Focus purely on relevance (equivalent to regular search)
            - 0.5: Equal balance between relevance and diversity
        **kwargs : Any
            Additional arguments for the search, such as:
            - metadata_callback: A function that takes (path, index) and returns
              additional metadata to include in the Document.
            
        Returns
        -------
        List[Document]
            A list of Document objects containing diverse but relevant images.
            
        Examples
        --------
        >>> from anomaly_reaper.infrastructure.vector_database.images_store import ImagesVectorStore
        >>> 
        >>> # Assuming 'store' is an already initialized ImagesVectorStore
        >>> 
        >>> # Search with default balance between relevance and diversity
        >>> results = store.max_marginal_relevance_search(
        ...     "data/reference_image.png", 
        ...     k=5, 
        ...     fetch_k=10, 
        ...     lambda_mult=0.5
        ... )
        >>> 
        >>> # Print the diverse results
        >>> for doc in results:
        ...     print(f"Diverse image: {doc.page_content}")
        
        Notes
        -----
        - If the query image is not found in the store and no embedding_function is
          provided, a ValueError will be raised.
        - Setting lambda_mult closer to 1 will return results more similar to regular
          similarity search.
        - Higher values of fetch_k create a larger pool of candidates for diversity
          selection, which may improve diversity but will take longer to compute.
        """
        # Convert query to embedding
        query_embedding = None
        
        # Case 1: Query is a path that exists in our image_paths_list
        if query in self.image_paths_list:
            # Get the index of the path
            idx = self.image_paths_list.index(query)
            # Get the embedding
            query_embedding = self.embeddings_df[idx]
        
        # Case 2: Query is a path to an image we haven't seen, but we have an embedding function
        elif self.embedding_function is not None:
            query_embedding = self.embedding_function.embed_query(query)
        
        # If we couldn't get an embedding, raise an error
        if query_embedding is None:
            raise ValueError(
                "Could not generate embedding for query. "
                "Either the query must be a known image path or "
                "embedding_function must be provided."
            )
        
        # Now we can do the vector-based MMR search
        return self.max_marginal_relevance_search_by_vector(
            query_embedding, k, fetch_k, lambda_mult, **kwargs
        )
    
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for diverse but relevant images using a vector embedding directly.
        
        This method implements Maximal Marginal Relevance (MMR) search to balance
        relevance to the query with diversity among selected results. It directly 
        uses a query embedding vector rather than a query image path.
        
        Parameters
        ----------
        embedding : List[float]
            The embedding vector to search with. Must have the same dimensions
            as the embeddings stored in the vector store.
        k : int, default=4
            Number of diverse documents to return.
        fetch_k : int, default=20
            Number of documents to initially fetch before applying diversity filtering.
            Should be greater than or equal to k.
        lambda_mult : float, default=0.5
            Balance factor between relevance and diversity. Ranges from 0 to 1:
            - 0: Focus purely on diversity (may return less relevant results)
            - 1: Focus purely on relevance (equivalent to regular search)
            - 0.5: Equal balance between relevance and diversity
        **kwargs : Any
            Additional arguments for the search, such as:
            - metadata_callback: A function that takes (path, index) and returns
              additional metadata to include in the Document.
            
        Returns
        -------
        List[Document]
            A list of Document objects containing diverse but relevant images.
            
        Examples
        --------
        >>> from anomaly_reaper.infrastructure.vector_database.images_store import ImagesVectorStore
        >>> import numpy as np
        >>> 
        >>> # Assuming 'store' is an already initialized ImagesVectorStore
        >>> # And we have a query embedding from somewhere
        >>> query_embedding = np.random.rand(512).tolist()  # Example with random vector
        >>> 
        >>> # Search with diversity consideration
        >>> results = store.max_marginal_relevance_search_by_vector(
        ...     query_embedding, 
        ...     k=5, 
        ...     fetch_k=10, 
        ...     lambda_mult=0.3  # Higher weight on diversity
        ... )
        >>> 
        >>> # Print the diverse results
        >>> for doc in results:
        ...     print(f"Diverse image: {doc.page_content}")
        
        Notes
        -----
        - The query embedding must have the same dimension as the embeddings in the store.
        - Higher values of lambda_mult give more importance to relevance, while lower 
          values prioritize diversity.
        - The algorithm first retrieves the top fetch_k most similar documents, then 
          iteratively selects documents to maximize the MMR criterion.
        - For small datasets, fetch_k will be automatically reduced to the dataset size.
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Convert to numpy for efficient calculations
        query_embedding = np.array(embedding).reshape(1, -1)
        embeddings_array = np.array(self.embeddings_df)
        
        # Get top fetch_k results first
        similarities = cosine_similarity(query_embedding, embeddings_array)[0]
        
        # Limit fetch_k to avoid index errors with smaller datasets
        if fetch_k > len(similarities):
            fetch_k = len(similarities)
        
        # Get indices of top fetch_k most similar embeddings
        top_indices = np.argsort(similarities)[-fetch_k:][::-1]
        
        # If k is greater than fetch_k, reduce k
        if k > fetch_k:
            k = fetch_k
        
        # If we only need one result or have only one result, return it
        if k == 1 or len(top_indices) == 1:
            idx = top_indices[0]
            embedding_vector = self.embeddings_df[idx]
            embedding_key = tuple(embedding_vector)
            path = self.image_paths_list[idx]
            
            # Start with stored metadata if available
            if embedding_key in self._embeddings_to_metadata:
                metadata = self._embeddings_to_metadata[embedding_key].copy()
            else:
                metadata = {}
            
            # Only include source if not already in metadata
            if "source" not in metadata:
                metadata["source"] = path
            
            # Use metadata_callback if provided
            metadata_callback = kwargs.get("metadata_callback")
            if metadata_callback and callable(metadata_callback):
                try:
                    user_metadata = metadata_callback(path, idx)
                    if user_metadata and isinstance(user_metadata, dict):
                        metadata.update(user_metadata)
                except Exception as e:
                    print(f"Error in metadata callback: {e}")
            
            return [Document(page_content=path, metadata=metadata)]
        
        # Extract relevant embeddings for top indices
        top_embeddings = embeddings_array[top_indices]
        
        # Initialize with first (most relevant) result
        selected_indices = [top_indices[0]]
        selected_embeddings = top_embeddings[0:1]
        
        # Iteratively build the set of selected documents
        while len(selected_indices) < k:
            best_score = -np.inf
            best_index = -1
            
            # For each candidate document
            for i, idx in enumerate(top_indices):
                if idx in selected_indices:
                    continue
                
                # Calculate relevance (similarity to query)
                relevance = similarities[idx]
                
                # Calculate diversity (negative maximum similarity to already selected docs)
                candidate_embedding = embeddings_array[idx].reshape(1, -1)
                diversity_scores = cosine_similarity(candidate_embedding, selected_embeddings)[0]
                diversity_penalty = np.max(diversity_scores)
                
                # Calculate MMR score: balance between relevance and diversity
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = idx
            
            if best_index == -1:  # No more diverse results found
                break
                
            # Add the best candidate to our selected set
            selected_indices.append(best_index)
            selected_embeddings = np.vstack([selected_embeddings, embeddings_array[best_index]])
        
        # Get metadata_callback if provided
        metadata_callback = kwargs.get("metadata_callback")
        
        # Convert to Documents
        results = []
        for idx in selected_indices:
            # Get embedding and path
            embedding_vector = self.embeddings_df[idx]
            embedding_key = tuple(embedding_vector)
            path = self.image_paths_list[idx]
            
            # Start with stored metadata if available
            if embedding_key in self._embeddings_to_metadata:
                metadata = self._embeddings_to_metadata[embedding_key].copy()
            else:
                metadata = {}
            
            # Only include source if not already in metadata
            if "source" not in metadata:
                metadata["source"] = path
            
            # Use callback for additional metadata if provided
            if metadata_callback and callable(metadata_callback):
                try:
                    user_metadata = metadata_callback(path, idx)
                    if user_metadata and isinstance(user_metadata, dict):
                        metadata.update(user_metadata)
                except Exception as e:
                    print(f"Error in metadata callback: {e}")
            
            # Create document
            doc = Document(page_content=path, metadata=metadata)
            results.append(doc)
        
        return results
    
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        Select the relevance score function based on the similarity method.
        For vector stores using cosine similarity, this is the cosine score.
        
        Returns:
            Function to convert raw similarity score to relevance score
        """
        return self._cosine_relevance_score_fn
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "ImagesVectorStore":
        """Create an ImagesVectorStore from image paths, URLs or base64 strings.
        
        This method generates embeddings for a list of image paths or encoded images
        and creates a vector store using those embeddings.
        
        Parameters
        ----------
        texts : List[str]
            List of image paths, URLs, or base64-encoded image strings.
        embedding : Embeddings
            The embedding function to use for generating vector representations of images.
        metadatas : Optional[List[dict]], default=None
            Optional metadata for each image. If provided, must be the same length as texts.
            Each dictionary can contain additional information about the corresponding image.
        ids : Optional[List[str]], default=None
            Optional IDs for each image. If provided, must be the same length as texts.
        **kwargs : Any
            Additional arguments to pass to the ImagesVectorStore constructor.
            
        Returns
        -------
        ImagesVectorStore
            An initialized vector store containing the generated embeddings and image paths.
            
        Examples
        --------
        >>> from anomaly_reaper.infrastructure.vector_database.images_store import ImagesVectorStore
        >>> from custom_embeddings import ImageEmbeddings  # Custom embedding function
        >>> 
        >>> # List of image paths
        >>> image_paths = ["images/img1.png", "images/img2.png", "images/img3.png"]
        >>> 
        >>> # Optional metadata
        >>> metadata = [
        ...     {"category": "nature", "tags": ["sunset", "beach"]},
        ...     {"category": "urban", "tags": ["building", "city"]},
        ...     {"category": "portrait", "tags": ["person", "face"]}
        ... ]
        >>> 
        >>> # Create embeddings function
        >>> embedding_function = ImageEmbeddings()
        >>> 
        >>> # Create vector store
        >>> store = ImagesVectorStore.from_texts(
        ...     texts=image_paths,
        ...     embedding=embedding_function,
        ...     metadatas=metadata
        ... )
        >>> 
        >>> # Now search with a query image
        >>> results = store.similarity_search("images/query_image.png", k=2)
        
        Notes
        -----
        - This method handles the embedding generation internally, which may be 
          resource-intensive for large collections of images.
        - The embedding function should be designed to work with image data.
        - Metadata is optional but recommended for storing additional information 
          about each image that can be useful during retrieval.
        """
        # Generate embeddings
        embeddings = embedding.embed_documents(texts)
        
        # Create instance
        store = cls(
            embeddings_df=embeddings,
            image_paths_list=list(texts),
            embedding_function=embedding,
            **kwargs
        )
        
        # Store metadata if provided
        if metadatas:
            for i, (path, emb) in enumerate(zip(texts, embeddings)):
                if i < len(metadatas):
                    # Use embedding as key
                    embedding_key = tuple(emb)
                    store._embeddings_to_metadata[embedding_key] = metadatas[i]
        
        return store
    
    @classmethod
    def from_numpy(
        cls,
        embeddings_array: Any,  # numpy.ndarray
        image_paths_list: List[str],
        embedding_function: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> "ImagesVectorStore":
        """Create an ImagesVectorStore from a numpy array of embeddings.
        
        This method allows you to create a vector store directly from pre-computed
        embeddings stored in a numpy array, which is useful when working with 
        existing image embedding models or when transferring embeddings from
        another system.
        
        Parameters
        ----------
        embeddings_array : numpy.ndarray
            NumPy array of embeddings with shape (n_samples, n_features).
            Each row represents an embedding vector for one image.
        image_paths_list : List[str]
            List of corresponding image paths or URIs. Must have the same length
            as the first dimension of embeddings_array.
        embedding_function : Optional[Embeddings], default=None
            Optional embedding function to use for new images that may be added later.
            Required if you plan to use add_texts() or similar functions.
        **kwargs : Any
            Additional arguments to pass to the ImagesVectorStore constructor.
            
        Returns
        -------
        ImagesVectorStore
            An initialized vector store containing the provided embeddings and image paths.
            
        Examples
        --------
        >>> import numpy as np
        >>> from anomaly_reaper.infrastructure.vector_database.images_store import ImagesVectorStore
        >>> 
        >>> # Example with random embeddings (in practice, these would be real image embeddings)
        >>> embeddings = np.random.rand(100, 512)  # 100 images with 512-dim embeddings
        >>> image_paths = [f"images/img_{i}.png" for i in range(100)]
        >>> 
        >>> # Create vector store
        >>> store = ImagesVectorStore.from_numpy(
        ...     embeddings_array=embeddings,
        ...     image_paths_list=image_paths
        ... )
        >>> 
        >>> # Search with a sample embedding vector
        >>> query_embedding = embeddings[0]  # Use the first embedding as a query
        >>> results = store.similarity_search_by_vector(query_embedding, k=5)
        >>> print(f"Found {len(results)} similar images")
        
        Notes
        -----
        - The embeddings_array must be a 2D numpy array.
        - The number of image paths must match the number of embedding vectors.
        - If you plan to add new images to the store later, you should provide an 
          embedding_function that can generate compatible embeddings.
        """
        # Convert numpy array to list of lists
        embeddings_list = embeddings_array.tolist()
        
        return cls(
            embeddings_df=embeddings_list,
            image_paths_list=image_paths_list,
            embedding_function=embedding_function,
            **kwargs
        )
    
    @classmethod
    def from_existing_index(
        cls,
        embeddings_path: str,
        image_paths_path: str,
        embedding_function: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> "ImagesVectorStore":
        """Create an ImagesVectorStore from existing CSV files.
        
        This method loads image embeddings and paths from CSV files, allowing you
        to persist and reload your vector store across sessions without having to
        recompute embeddings.
        
        Parameters
        ----------
        embeddings_path : str
            Path to CSV file containing image embeddings. The CSV should contain
            numeric embedding values, with each row representing one image embedding.
            No header is expected in the CSV file.
        image_paths_path : str
            Path to CSV file containing image paths or URIs. The CSV should have a 
            single column of path strings, with each row corresponding to the 
            same-indexed row in the embeddings CSV.
        embedding_function : Optional[Embeddings], default=None
            Optional embedding function to use for new images that may be added later.
            Required if you plan to use add_texts() or similar functions.
        **kwargs : Any
            Additional arguments to pass to the ImagesVectorStore constructor.
            
        Returns
        -------
        ImagesVectorStore
            An initialized vector store containing the embeddings and image paths
            loaded from the provided CSV files.
            
        Examples
        --------
        >>> from anomaly_reaper.infrastructure.vector_database.images_store import ImagesVectorStore
        >>> 
        >>> # Load from existing CSV files
        >>> store = ImagesVectorStore.from_existing_index(
        ...     embeddings_path="data/image_embeddings.csv",
        ...     image_paths_path="data/image_paths.csv"
        ... )
        >>> 
        >>> # Now use the loaded store for searching
        >>> results = store.similarity_search("images/reference_image.png", k=3)
        >>> for doc in results:
        ...     print(f"Similar image: {doc.page_content}")
        
        Notes
        -----
        - The CSV files should have the same number of rows.
        - The embeddings CSV should contain only the numeric values of embeddings.
        - The image paths CSV should have a single column with no header.
        - If embeddings have different dimensions, make sure they're all padded to
          the same length before saving to CSV.
        - For large datasets, consider using from_numpy with compressed numpy files
          (.npz) for better performance.
        """
        
        # Load embeddings and image paths from CSV
        embeddings_df = pd.read_csv(embeddings_path)
        image_paths_df = pd.read_csv(image_paths_path)
        
        # Convert to appropriate format
        embeddings_list = embeddings_df.values.tolist()
        image_paths_list = image_paths_df.iloc[:, 0].tolist()
        
        return cls(
            embeddings_df=embeddings_list,
            image_paths_list=image_paths_list,
            embedding_function=embedding_function,
            **kwargs
        )
    
    @staticmethod
    def _cosine_relevance_score_fn(score: float) -> float:
        """Convert cosine similarity score to relevance score.
        
        This function is used to transform the raw similarity score
        into a normalized relevance score. For cosine similarity,
        this is simply the raw score as it's already within a suitable
        range (-1 to 1).
        
        Parameters
        ----------
        score : float
            The raw similarity score (cosine similarity)
            
        Returns
        -------
        float
            The transformed relevance score, same as input for cosine similarity
            
        Notes
        -----
        - Cosine similarity scores range from -1 (complete dissimilarity) 
          to 1 (perfect similarity).
        - This method is called internally by the vector store when 
          ranking search results.
        """
        return score

