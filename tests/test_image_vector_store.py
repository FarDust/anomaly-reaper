"""
Tests for Image Vector Store with mock embedding function.

These tests verify the functionality of the ImagesVectorStore class
using a mock embedding function instead of the actual VertexAI API.
"""

import os
import pytest
import tempfile
import numpy as np
import pandas as pd
import base64
from pathlib import Path
from PIL import Image, ImageDraw
import io
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from anomaly_reaper.infrastructure.vector_database.images_store import ImagesVectorStore
from anomaly_reaper.infrastructure.vector_database.image_embeddings import VertexAIMultimodalImageEmbeddings


# Test constants
B64_ENCODED_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAABOFBMVEU4SD5AUEVWZVZLYFNcaVpVYlVSXlBNXFBLWk5TXU5UZFRaallBWUxFX0xJXE9TZldNYE9EXE1YaFZCU0g6TEJLZVImeysEtgAWxwUizQBVYFExQDRBVkpSaFlOd1E7YkUgkSIUlxcfyA0VoBcr0Q1n3zMAvwARmhVU2ykRygOY6UdU3A9P2hlE0xR04y0tQjwwRz5fb11SYlFYblpZc1xPY1Q9jjo4az5MbFYvgzMJqwpEc1A3Ujc5VDcFoAcnjS0huhIgwQgmoScDuQEtzA8+zhyn7TpCzwxw4h2K5yg61Qgl0gJF2Q1q4gsBvAAuOC8VJSAjNCtSZFFWbF1Sh1NfdmVMl05khWg6cUQchyIqtBoZrxY20BNCZEtD1Chl3ieF5TvC8WBd2yINwQOv7V+v7V618Dy48F3ii7xDAAAACnRSTlP9/ff+9/f39/f3EdmZOQAAAQRJREFUGNMNzOWWglAUQOGjjjEXuBcQCQEVwcbuxO6Ybqfj/d9g/LvXtzZAsShotMbRdNDlEgQAoVSiOY45HhmMtVODINBMVIylUo9Npo6QFzBuPRjD0WwhDcQQIX7w1LupfT6b/vyadESWBIBjnL09/ykU8lNp0KTcQEfNzOr7L5dRZFkSWT+Q7vNr4uN3l7OU3cKpByAUeZLXq+3GLuTeZjHkA71mJhUr8b600y+T9umhV9tSMp/NzjdKvBMpn4FHbxnJzHK7luOjRhm5AbMkMo4n0tZ02KhS/EmQ69v7u7FpODc1iqe8AOqlKvT6vT5zVUFhdA7Bw4WKKyzLU4Tiw2HfPzrPJ1QyTWRpAAAAAElFTkSuQmCC"


class MockEmbeddings(Embeddings):
    """Mock embedding class for testing."""
    
    def __init__(self, dimension=1408):
        """Initialize with a specified embedding dimension."""
        self.dimension = dimension
        self.embed_count = 0
        self.cache = {}  # Cache to ensure consistent results for the same input
        np.random.seed(42)  # For consistent test results
    
    def embed_documents(self, texts):
        """Generate mock embeddings for documents."""
        self.embed_count += len(texts)
        
        results = []
        for text in texts:
            # Use hash of the text to create a consistent but unique embedding
            if text not in self.cache:
                # Create a deterministic but unique embedding for this text
                np.random.seed(hash(text) % 10000)
                self.cache[text] = np.random.rand(self.dimension).tolist()
            
            results.append(self.cache[text])
            
        return results
    
    def embed_query(self, text):
        """Generate a mock embedding for a query."""
        self.embed_count += 1
        
        # For query, use the same caching mechanism to ensure consistent results
        if text not in self.cache:
            np.random.seed(hash(text) % 10000)
            self.cache[text] = np.random.rand(self.dimension).tolist()
            
        return self.cache[text]


@pytest.fixture
def test_image_file():
    """Create a small test image file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        # Create a small test image
        img = Image.new('RGB', (50, 50), color='white')
        
        # Draw something on the image
        draw = ImageDraw.Draw(img)
        draw.rectangle([(10, 10), (40, 40)], fill='black')
        
        # Save the image to the temporary file
        img.save(tmp.name)
        
        # Return the path to the file
        path = tmp.name
    
    yield path
    
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def test_image_b64():
    """Generate a base64-encoded test image."""
    # Create a small test image
    img = Image.new('RGB', (50, 50), color='white')
    
    # Draw something on the image
    draw = ImageDraw.Draw(img)
    draw.rectangle([(15, 15), (35, 35)], fill='red')
    
    # Convert the image to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    b64_image = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/png;base64,{b64_image}"


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings function."""
    return MockEmbeddings()


@pytest.fixture
def base_vector_store(mock_embeddings, test_image_file, test_image_b64):
    """Create a vector store with test images."""
    # Create test images
    test_images = [
        test_image_file,
        test_image_b64,
        B64_ENCODED_IMAGE
    ]
    
    # Create the vector store
    vector_store = ImagesVectorStore.from_texts(
        texts=test_images,
        embedding=mock_embeddings,
        metadatas=[
            {"source": "file", "description": "Test rectangle image"},
            {"source": "base64", "description": "Red square image"},
            {"source": "base64", "description": "Icon image"}
        ]
    )
    
    return vector_store


@pytest.fixture
def mock_embeddings_data():
    """Create mock embeddings data for testing."""
    # Number of sample embeddings and dimensions
    num_samples = 5
    dimension = 1408
    
    # Generate consistent random embeddings
    np.random.seed(42)
    embeddings = np.random.rand(num_samples, dimension)
    
    # Generate mock image paths
    paths = [f"test_image_{i}.png" for i in range(num_samples)]
    
    return embeddings, paths


@pytest.fixture
def temp_csv_files(mock_embeddings_data):
    """Create temporary CSV files with mock data for testing from_existing_index."""
    mock_embeddings, mock_paths = mock_embeddings_data
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create embeddings CSV
        embeddings_path = os.path.join(temp_dir, "test_embeddings.csv")
        paths_path = os.path.join(temp_dir, "test_paths.csv")
        
        # Save to CSV
        np.savetxt(embeddings_path, mock_embeddings, delimiter=',', 
                  header=','.join([f'f{i}' for i in range(mock_embeddings.shape[1])]))
        pd.DataFrame(mock_paths).to_csv(paths_path, index=False, header=['path'])
        
        yield embeddings_path, paths_path


@pytest.fixture
def mock_vertex_embedding_response():
    """Create a mock embedding response similar to what VertexAI would return."""
    # Create a mock response with image_embedding attribute
    mock_response = MagicMock()
    # Create a deterministic random embedding
    np.random.seed(42)
    mock_response.image_embedding = np.random.rand(1408).tolist()
    return mock_response


@pytest.fixture
def mock_vertex_embedding_model(mock_vertex_embedding_response):
    """Create a mock for the VertexAI embedding model."""
    mock_model = MagicMock()
    mock_model.get_embeddings.return_value = mock_vertex_embedding_response
    return mock_model


@pytest.fixture
def mocked_vertex_embedding():
    """Create a mocked VertexAIMultimodalImageEmbeddings instance."""
    # Patch the MultiModalEmbeddingModel
    with patch('vertexai.vision_models.MultiModalEmbeddingModel') as mock_model_class:
        # Setup the mock
        mock_model = MagicMock()
        
        # Create deterministic embeddings with small variations for different inputs
        def mock_get_embeddings(image, dimension, contextual_text):
            # Use image object id as seed for reproducible but unique embeddings
            seed = hash(str(image)) % 10000
            np.random.seed(seed)
            mock_response = MagicMock()
            mock_response.image_embedding = np.random.rand(dimension).tolist()
            return mock_response
        
        mock_model.get_embeddings.side_effect = mock_get_embeddings
        
        # Set up the from_pretrained to return our mock model
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create the embedding function with the mocked class
        embedding_func = VertexAIMultimodalImageEmbeddings()
        
        # Manually set the private attribute to avoid actual API calls
        embedding_func._embedding_model = mock_model
        
        yield embedding_func


class TestImagesVectorStoreCreation:
    """Test the various creation methods for ImagesVectorStore."""
    
    def test_from_texts_creation(self, mock_embeddings, test_image_file, test_image_b64):
        """Test creating a vector store from image paths and base64 strings."""
        # Create vector store
        vector_store = ImagesVectorStore.from_texts(
            texts=[test_image_file, test_image_b64, B64_ENCODED_IMAGE],
            embedding=mock_embeddings,
            metadatas=[
                {"test": "metadata1"},
                {"test": "metadata2"},
                {"test": "metadata3"}
            ]
        )
        
        # Check the store
        assert len(vector_store.embeddings_df) == 3
        assert len(vector_store.image_paths_list) == 3
        
        # Check paths were stored correctly
        assert test_image_file in vector_store.image_paths_list
        assert test_image_b64 in vector_store.image_paths_list
        assert B64_ENCODED_IMAGE in vector_store.image_paths_list
    
    def test_from_numpy_creation(self, mock_embeddings):
        """Test creating a vector store from numpy arrays."""
        # Create test data
        num_samples = 3
        embeddings = np.random.random((num_samples, 1408))
        paths = [f"path_{i}.jpg" for i in range(num_samples)]
        
        # Create vector store
        vector_store = ImagesVectorStore.from_numpy(
            embeddings_array=embeddings,
            image_paths_list=paths,
            embedding_function=mock_embeddings
        )
        
        # Check the store
        assert len(vector_store.embeddings_df) == num_samples
        assert len(vector_store.image_paths_list) == num_samples
        
        # Check values
        for i, path in enumerate(paths):
            assert path in vector_store.image_paths_list
    
    def test_from_existing_index(self, mock_embeddings, temp_csv_files):
        """Test creating a vector store from existing CSV files."""
        # Get temp file paths
        embeddings_path, paths_path = temp_csv_files
        
        # Create vector store
        vector_store = ImagesVectorStore.from_existing_index(
            embeddings_path=embeddings_path,
            image_paths_path=paths_path,
            embedding_function=mock_embeddings
        )
        
        # Check the store
        assert len(vector_store.embeddings_df) == 5  # Based on mock_embeddings_data fixture
        assert len(vector_store.image_paths_list) == 5


class TestImagesVectorStoreOperations:
    """Test operations on the ImagesVectorStore class."""
    
    def test_add_texts(self, base_vector_store, test_image_b64):
        """Test adding new images to the vector store."""
        # Count initial items
        initial_count = len(base_vector_store.embeddings_df)
        
        # Add a new image
        new_image_b64 = test_image_b64.replace("red", "blue")  # Just to make it a bit different
        ids = base_vector_store.add_texts(
            texts=[new_image_b64],
            metadatas=[{"source": "test_add", "color": "blue"}]
        )
        
        # Check result
        assert len(ids) == 1
        assert len(base_vector_store.embeddings_df) == initial_count + 1
        assert new_image_b64 in base_vector_store.image_paths_list
    
    def test_similarity_search(self, base_vector_store, test_image_file):
        """Test similarity search functionality."""
        # Perform a similarity search
        results = base_vector_store.similarity_search(
            query=test_image_file,
            k=2
        )
        
        # Check results
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)
        assert all("source" in doc.metadata for doc in results)
    
    def test_similarity_search_with_score(self, base_vector_store, test_image_file):
        """Test similarity search with scores."""
        # Perform a similarity search with scores
        results = base_vector_store.similarity_search_with_score(
            query=test_image_file,
            k=2
        )
        
        # Check results
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        
        # First result should have highest score (closest to 1)
        assert results[0][1] > 0
    
    def test_similarity_search_by_vector(self, base_vector_store, mock_embeddings):
        """Test similarity search using a vector directly."""
        # Generate a random query embedding
        query_embedding = mock_embeddings.embed_query("dummy_query")
        
        # Perform a similarity search by vector
        results = base_vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=2
        )
        
        # Check results
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)
    
    def test_max_marginal_relevance_search(self, base_vector_store, test_image_file):
        """Test max marginal relevance search for diversity."""
        # Add a few more similar images
        similar_images = []
        
        # Create similar but slightly different images
        for i in range(3):
            # Create a mock image path
            temp_file = f"{test_image_file}_{i}.png"
            similar_images.append(temp_file)
        
        # Add to vector store
        base_vector_store.add_texts(
            texts=similar_images,
            metadatas=[{"source": "similar", "index": i} for i in range(3)]
        )
        
        # Perform MMR search
        results_regular = base_vector_store.similarity_search(
            query=test_image_file,
            k=4
        )
        
        results_mmr = base_vector_store.max_marginal_relevance_search(
            query=test_image_file,
            k=4,
            lambda_mult=0.3  # Emphasize diversity
        )
        
        # Check results
        assert len(results_regular) == 4
        assert len(results_mmr) == 4
    
    def test_delete(self, base_vector_store, test_image_file):
        """Test deleting images from the vector store."""
        # Get initial count
        initial_count = len(base_vector_store.embeddings_df)
        
        # Delete by path pattern
        success = base_vector_store.delete(path_pattern=os.path.basename(test_image_file))
        
        # Check results
        assert success is True
        assert len(base_vector_store.embeddings_df) < initial_count
        assert test_image_file not in base_vector_store.image_paths_list
    
    def test_metadata_persistence(self, mock_embeddings, test_image_file, test_image_b64):
        """Test that metadata is correctly preserved across different operations."""
        # Create test images with specific metadata
        test_images = [
            test_image_file,
            test_image_b64,
            B64_ENCODED_IMAGE
        ]
        
        # Create detailed metadata with various data types
        metadata_list = [
            {
                "source": "file", 
                "description": "Test rectangle image",
                "numeric_value": 123,
                "tags": ["test", "rectangle"],
                "is_valid": True,
                "nested": {"key1": "value1", "key2": 2}
            },
            {
                "source": "base64", 
                "description": "Red square image",
                "numeric_value": 456,
                "tags": ["test", "square", "red"],
                "is_valid": True,
                "nested": {"key1": "value2", "key2": 3}
            },
            {
                "source": "base64", 
                "description": "Icon image",
                "numeric_value": 789,
                "tags": ["test", "icon"],
                "is_valid": False,
                "nested": {"key1": "value3", "key2": 4}
            }
        ]
        
        # Create the vector store
        vector_store = ImagesVectorStore.from_texts(
            texts=test_images,
            embedding=mock_embeddings,
            metadatas=metadata_list
        )
        
        # 1. Test metadata is preserved in basic similarity search
        results = vector_store.similarity_search(
            query=test_image_file,
            k=3
        )
        
        # Verify all original metadata is present
        for doc in results:
            assert "source" in doc.metadata
            assert "description" in doc.metadata
            assert "numeric_value" in doc.metadata
            assert "tags" in doc.metadata
            assert "is_valid" in doc.metadata
            assert "nested" in doc.metadata
            
            # Check data types are preserved
            assert isinstance(doc.metadata["numeric_value"], int)
            assert isinstance(doc.metadata["tags"], list)
            assert isinstance(doc.metadata["is_valid"], bool)
            assert isinstance(doc.metadata["nested"], dict)
        
        # 2. Test metadata is preserved in similarity search with score
        results_with_score = vector_store.similarity_search_with_score(
            query=test_image_file,
            k=3
        )
        
        for doc, _ in results_with_score:
            assert "source" in doc.metadata
            assert "description" in doc.metadata
            assert "numeric_value" in doc.metadata
            assert isinstance(doc.metadata["numeric_value"], int)
            assert isinstance(doc.metadata["tags"], list)
        
        # 3. Test metadata is preserved in MMR search
        mmr_results = vector_store.max_marginal_relevance_search(
            query=test_image_file,
            k=3,
            lambda_mult=0.5
        )
        
        for doc in mmr_results:
            assert "source" in doc.metadata
            assert "description" in doc.metadata
            assert "nested" in doc.metadata
            assert isinstance(doc.metadata["nested"], dict)
            assert "key1" in doc.metadata["nested"]
            assert "key2" in doc.metadata["nested"]
        
        # 4. Test metadata is preserved after adding new items
        new_image = test_image_b64.replace("red", "blue")
        new_metadata = {
            "source": "new_test",
            "description": "New blue image",
            "numeric_value": 999,
            "tags": ["new", "blue"],
            "is_valid": True,
            "nested": {"key1": "new_value", "key2": 5}
        }
        
        vector_store.add_texts(
            texts=[new_image],
            metadatas=[new_metadata]
        )
        
        # Search to get the new item
        new_results = vector_store.similarity_search(
            query=new_image,
            k=1
        )
        
        # Verify the new metadata is correct
        assert len(new_results) == 1
        new_doc = new_results[0]
        assert new_doc.metadata["source"] == "new_test"
        assert new_doc.metadata["description"] == "New blue image"
        assert new_doc.metadata["numeric_value"] == 999
        assert "new" in new_doc.metadata["tags"]
        assert "blue" in new_doc.metadata["tags"]
        assert new_doc.metadata["is_valid"] is True
        assert new_doc.metadata["nested"]["key1"] == "new_value"
        assert new_doc.metadata["nested"]["key2"] == 5
        
        # 5. Test metadata is preserved in vector search
        # Get an embedding to search with
        query_embedding = mock_embeddings.embed_query(test_image_file)
        
        vector_results = vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=3
        )
        
        for doc in vector_results:
            assert "source" in doc.metadata
            assert "description" in doc.metadata
            assert isinstance(doc.metadata["numeric_value"], int)
            assert isinstance(doc.metadata["tags"], list)
            assert isinstance(doc.metadata["nested"], dict)
        
        # 6. Test metadata is preserved after deletion operation
        # Delete one image
        success = vector_store.delete(path_pattern=os.path.basename(test_image_file))
        assert success is True
        
        # Check remaining metadata is intact
        remaining_results = vector_store.similarity_search(
            query=test_image_b64,
            k=3
        )
        
        for doc in remaining_results:
            # The file was deleted, so verify we only see the other items
            assert doc.page_content != test_image_file
            assert "source" in doc.metadata
            assert "description" in doc.metadata
            assert "numeric_value" in doc.metadata
            assert "tags" in doc.metadata
            assert "is_valid" in doc.metadata
            assert "nested" in doc.metadata
            assert isinstance(doc.metadata["nested"], dict)


class TestVertexAIIntegration:
    """Test the integration between VertexAIMultimodalImageEmbeddings and ImagesVectorStore."""
    
    @patch('vertexai.vision_models.MultiModalEmbeddingModel')
    @patch('vertexai.vision_models.Image')
    def test_vertex_embeddings_basic(self, mock_vertex_image, mock_model_class, test_image_file):
        """Test basic VertexAIMultimodalImageEmbeddings functionality with mocks."""
        # Set up mock model return
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.image_embedding = np.random.rand(1408).tolist()
        mock_model.get_embeddings.return_value = mock_response
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create embedding function
        embedding_function = VertexAIMultimodalImageEmbeddings()
        
        # Manually set model to avoid actual API calls
        embedding_function._embedding_model = mock_model
        
        # Generate an embedding
        embedding = embedding_function.embed_query(test_image_file)
        
        # Verify
        assert len(embedding) == 1408
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)
        
        # Verify the mock was called properly
        mock_model.get_embeddings.assert_called()
    
    @patch('vertexai.vision_models.MultiModalEmbeddingModel')
    @patch('vertexai.vision_models.Image')
    def test_vertex_embeddings_with_vector_store(self, mock_vertex_image, mock_model_class, test_image_file, test_image_b64):
        """Test integration between VertexAIMultimodalImageEmbeddings and ImagesVectorStore."""
        # Set up mock model return
        mock_model = MagicMock()
        
        # Configure mock to return different embeddings for different inputs
        def mock_get_embeddings(image, dimension, contextual_text):
            # Use the image hash to seed the random generator
            seed = hash(str(image)) % 10000
            np.random.seed(seed)
            mock_response = MagicMock()
            mock_response.image_embedding = np.random.rand(dimension).tolist()
            return mock_response
        
        mock_model.get_embeddings.side_effect = mock_get_embeddings
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create embedding function
        embedding_function = VertexAIMultimodalImageEmbeddings()
        embedding_function._embedding_model = mock_model
        
        # Create test images
        test_images = [
            test_image_file,
            test_image_b64,
            B64_ENCODED_IMAGE
        ]
        
        # Create vector store using VertexAI embeddings
        vector_store = ImagesVectorStore.from_texts(
            texts=test_images,
            embedding=embedding_function,
            metadatas=[
                {"source": "file", "description": "Test rectangle image"},
                {"source": "base64", "description": "Red square image"},
                {"source": "base64", "description": "Icon image"}
            ]
        )
        
        # Verify vector store was created correctly
        assert len(vector_store.embeddings_df) == 3
        assert len(vector_store.image_paths_list) == 3
        
        # Test search
        results = vector_store.similarity_search(
            query=test_image_file,
            k=2
        )
        
        # Verify search results
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)
        
        # Verify the first result is related to our query
        assert "source" in results[0].metadata
        
        # Test integration of other search methods
        vector_results = vector_store.similarity_search_by_vector(
            embedding=embedding_function.embed_query(test_image_b64),
            k=2
        )
        assert len(vector_results) == 2
    
    def test_integrated_workflow(self, mocked_vertex_embedding, test_image_file, test_image_b64):
        """Test full workflow with mocked VertexAI embeddings."""
        # Create images list
        test_images = [
            test_image_file, 
            test_image_b64,
            B64_ENCODED_IMAGE
        ]
        
        # 1. Create vector store with VertexAI embeddings
        vector_store = ImagesVectorStore.from_texts(
            texts=test_images,
            embedding=mocked_vertex_embedding,
            metadatas=[
                {"source": "file", "description": "Test file"},
                {"source": "base64", "description": "Red square"},
                {"source": "base64", "description": "Icon"}
            ]
        )
        
        # 2. Add a new image
        new_image_b64 = test_image_b64.replace("red", "green")
        vector_store.add_texts(
            texts=[new_image_b64],
            metadatas=[{"source": "test_add", "color": "green"}]
        )
        
        # 3. Verify image was added
        assert len(vector_store.embeddings_df) == 4
        
        # 4. Search with original image
        results = vector_store.similarity_search(
            query=test_image_file,
            k=2
        )
        
        # 5. Verify search results
        assert len(results) == 2
        
        # 6. Try MMR search
        mmr_results = vector_store.max_marginal_relevance_search(
            query=test_image_file,
            k=3,
            lambda_mult=0.5
        )
        
        # 7. Verify MMR results
        assert len(mmr_results) == 3
        
        # 8. Delete an image
        success = vector_store.delete(path_pattern=os.path.basename(test_image_file))
        assert success is True
        assert len(vector_store.embeddings_df) == 3
        
        # 9. Try search with vector directly
        embedding = mocked_vertex_embedding.embed_query(B64_ENCODED_IMAGE)
        vector_results = vector_store.similarity_search_by_vector(
            embedding=embedding,
            k=2
        )
        
        # 10. Verify vector search results
        assert len(vector_results) == 2


def test_end_to_end_flow(mock_embeddings, test_image_file, test_image_b64):
    """Test end-to-end flow with a fresh vector store."""
    # 1. Create an empty vector store
    vector_store = ImagesVectorStore(
        embeddings_df=[],
        image_paths_list=[],
        embedding_function=mock_embeddings
    )
    
    # 2. Add some images
    vector_store.add_texts(
        texts=[test_image_file, test_image_b64, B64_ENCODED_IMAGE],
        metadatas=[
            {"source": "file", "description": "Test file"},
            {"source": "base64", "description": "Test base64"},
            {"source": "base64", "description": "Icon"}
        ]
    )
    
    # 3. Verify images were added
    assert len(vector_store.embeddings_df) == 3
    assert len(vector_store.image_paths_list) == 3
    
    # 4. Perform a similarity search
    results = vector_store.similarity_search(
        query=test_image_file,
        k=3
    )
    
    # 5. Check search results
    assert len(results) == 3
    assert "source" in results[0].metadata
    
    # 6. Try max marginal relevance search
    mmr_results = vector_store.max_marginal_relevance_search(
        query=test_image_file,
        k=3,
        lambda_mult=0.5
    )
    
    assert len(mmr_results) == 3
    
    # 7. Delete one image
    vector_store.delete(ids=["some_id_that_doesnt_exist"])  # Should not affect store
    assert len(vector_store.embeddings_df) == 3
    
    success = vector_store.delete(path_pattern=os.path.basename(test_image_file))
    assert success is True
    assert len(vector_store.embeddings_df) == 2
    
    # 8. Test retrieval after deletion
    results_after_delete = vector_store.similarity_search(
        query=test_image_b64,
        k=2
    )
    
    assert len(results_after_delete) == 2
    assert test_image_file not in [doc.page_content for doc in results_after_delete]




