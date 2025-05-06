from vertexai.vision_models import MultiModalEmbeddingModel
from vertexai.vision_models import Image as VertexImage
import base64
from logging import Logger, getLogger
from pathlib import Path
import re
from langchain_core.embeddings import Embeddings
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
)



class VertexAIMultimodalImageEmbeddings(Embeddings, BaseModel):
    """
    Generate multimodal embeddings for images using Google Vertex AI's MultiModalEmbedding model.
    
    This class implements LangChain's Embeddings interface to produce vector embeddings 
    for images using Google's multimodal embedding model. The embeddings represent the 
    semantic content of images and can be used for similarity search, clustering, 
    or anomaly detection.
    
    The class supports multiple image input formats:
    - Local file paths
    - Google Cloud Storage URIs (gs://)
    - Base64-encoded image strings
    
    Parameters
    ----------
    dimension : int, default=1408
        The dimensionality of the output embedding vector.
    context : str, default="deep space image with and anomaly detected by an automated system"
        Contextual text to guide the embedding model's understanding of the image.
        This helps provide domain context for more relevant embeddings.
    base64_match_pattern : str, default=r"data:image/([a-zA-Z0-9+]+);base64,(.+)"
        Regex pattern to match and extract base64-encoded image data.
    string_encoding : str, default="utf-8"
        Encoding format for string operations when handling base64 data.
        
    Attributes
    ----------
    _embedding_model : MultiModalEmbeddingModel
        The pre-trained Vertex AI multimodal embedding model instance.
    _logger : Logger
        Logger instance for tracking operations and debugging.
        
    Examples
    --------
    >>> # Initialize the embeddings model
    >>> image_embedder = VertexAIMultimodalImageEmbeddings()
    >>> 
    >>> # Embed a local image file
    >>> local_embedding = image_embedder.embed_query("/path/to/image.jpg")
    >>> print(f"Embedding dimension: {len(local_embedding)}")  # Should be 1408
    >>> 
    >>> # Embed an image from Google Cloud Storage
    >>> gcs_embedding = image_embedder.embed_query("gs://my-bucket/space-anomaly.png")
    >>> 
    >>> # Embed a base64-encoded image
    >>> b64_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAg..."
    >>> b64_embedding = image_embedder.embed_query(b64_image)
    >>> 
    >>> # Embed multiple images at once
    >>> image_paths = ["/path/to/image1.jpg", "gs://bucket/image2.png"]
    >>> embeddings = image_embedder.embed_documents(image_paths)
    >>> 
    >>> # Use with ImagesVectorStore for similarity search
    >>> vector_store = ImagesVectorStore.from_numpy(
    ...     embeddings_array=np.array(embeddings),
    ...     image_paths_list=image_paths,
    ...     embedding_function=image_embedder
    ... )
    >>> similar_images = vector_store.similarity_search("/path/to/query_image.jpg", k=5)
    >>> 
    >>> # With custom context for specific domain understanding
    >>> astronomy_embedder = VertexAIMultimodalImageEmbeddings(
    ...     context="astronomical image showing stellar formations and potential anomalies"
    ... )
    """

    dimension: int = Field(default=1408)
    context: str = Field(default="deep space image with and anomaly detected by an automated system")
    base64_match_pattern: str = Field(default=r"data:image/([a-zA-Z0-9+]+);base64,(.+)")
    string_encoding: str = Field(default="utf-8")

    _embedding_model: MultiModalEmbeddingModel = PrivateAttr(default=None)
    _logger: Logger = PrivateAttr(default=getLogger(__name__))

    # Adding model_config to allow arbitrary types
    model_config = {
        "arbitrary_types_allowed": True,
    }

    def _get_embedding_model(self) -> MultiModalEmbeddingModel:
        """Lazy loading of the embedding model to avoid recursion issues during initialization."""
        if self._embedding_model is None:
            self._embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        return self._embedding_model

    def _query_to_image(self, image_source: str) -> VertexImage:
        """
        Convert various image source formats to a VertexAI Image object.
        
        This helper method handles different input formats:
        - Base64-encoded image strings (matching base64_match_pattern)
        - Google Cloud Storage URIs (starting with "gs://")
        - Local file paths
        - Raw base64-encoded strings (fallback)
        
        Parameters
        ----------
        image_source : str
            The source of the image, which can be a file path, GCS URI, or base64 string
            
        Returns
        -------
        VertexImage
            A VertexAI Image object ready for embedding
            
        Notes
        -----
        If the image_source doesn't match any of the recognized patterns, it will be 
        treated as a raw base64-encoded string as a fallback mechanism.
        """
        b64_match = re.match(self.base64_match_pattern, image_source)
        image_source_as_path = Path(image_source)
        if b64_match:
            image = VertexImage(
                image_bytes=base64.decodebytes(b64_match.group(2).encode(encoding=self.string_encoding)),
            )
        elif image_source.startswith("gs://"):
            image = VertexImage(
                gcs_uri=image_source,
            )
        elif image_source and bool(image_source_as_path.parts) and image_source_as_path.is_file():

            image_bytes = image_source_as_path.read_bytes()
            image = VertexImage(
                image_bytes=image_bytes,
            )
        else:
            # pass the image directly as b64 input assuming well encoded data by the user
            image = VertexImage(
                image_bytes=base64.decodebytes(image_source.encode(encoding=self.string_encoding)),
            )
        return image
            


    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple images.
        
        Processes a list of image sources (file paths, GCS URIs, or base64 strings)
        and returns a list of embedding vectors, one for each image.
        
        Parameters
        ----------
        texts : list[str]
            A list of image sources. Each source can be a:
            - Local file path
            - Google Cloud Storage URI (gs://)
            - Base64-encoded image string
        
        Returns
        -------
        list[list[float]]
            A list of embedding vectors, where each vector is a list of floats
            representing the semantic content of the corresponding image.
            
        Examples
        --------
        >>> embedder = VertexAIMultimodalImageEmbeddings()
        >>> image_paths = [
        ...     "/path/to/image1.jpg", 
        ...     "gs://my-bucket/image2.png",
        ...     "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAg..."
        ... ]
        >>> embeddings = embedder.embed_documents(image_paths)
        >>> print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        """
        embeddings: list[list[float]] = []

        for image_source in texts:
            image = self._query_to_image(image_source)
            embedding = self.embed_query(image)
            embeddings.append(embedding)

        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single image.
        
        Converts the image source to a VertexAI Image object and generates
        an embedding vector using the multimodal embedding model.
        
        Parameters
        ----------
        text : str
            The image source, which can be a:
            - Local file path
            - Google Cloud Storage URI (gs://)
            - Base64-encoded image string
            - Or a VertexImage object (passed directly from embed_documents)
            
        Returns
        -------
        list[float]
            An embedding vector representing the semantic content of the image.
            The dimensionality is determined by the `dimension` parameter (default: 1408).
            
        Examples
        --------
        >>> embedder = VertexAIMultimodalImageEmbeddings()
        >>> embedding = embedder.embed_query("/path/to/space_image.jpg")
        >>> print(f"Embedding dimension: {len(embedding)}")
        >>> 
        >>> # Using a custom context for domain-specific understanding
        >>> embedder = VertexAIMultimodalImageEmbeddings(
        ...     context="telescope image from TESS satellite showing potential exoplanet transit"
        ... )
        >>> embedding = embedder.embed_query("gs://astronomy-data/exoplanet_candidate.fits")
        """
        image = self._query_to_image(text) if isinstance(text, str) else text

        # Use the lazy-loaded model
        embedding_model = self._get_embedding_model()
        embedding = embedding_model.get_embeddings(
            image=image,
            dimension=self.dimension,
            contextual_text=self.context,
        ).image_embedding
        
        return embedding