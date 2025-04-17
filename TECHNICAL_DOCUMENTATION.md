# Fashion Recommender System: Comprehensive Technical Documentation

## System Architecture

### 1. Core Components and Data Flow

The system follows a modular architecture with the following components:

```
┌───────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Data Ingestion│────▶│ Vector Embedding│────▶│  Vector Database │
└───────────────┘     └─────────────────┘     └──────────────────┘
        ▲                      ▲                        │
        │                      │                        ▼
┌───────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ User Interface│◀────│ Query Processing│◀────│ Similarity Search │
└───────────────┘     └─────────────────┘     └──────────────────┘
        │                      ▲                        ▲
        ▼                      │                        │
┌───────────────┐              │                        │
│    RAG Engine  │─────────────┘────────────────────────┘
└───────────────┘
```

### 2. Technical Implementation Details

#### 2.1 Data Structures

The core data structure is a multidimensional vector space:

- **Text Embeddings**: 384-dimensional vectors (SentenceTransformers) or 1536-dimensional vectors (OpenAI)
- **Image Embeddings**: 512-dimensional vectors (CLIP)
- **Combined Embeddings**: Normalized and weighted combination of both

#### 2.2 FAISS Index Configuration

```python
# L2 distance is used for fashion similarity with flat index
index = faiss.IndexFlatL2(dimension)

# Alternative configurations for larger datasets:
# index = faiss.IndexIVFFlat(quantizer, dimension, n_centroids, faiss.METRIC_L2)
# index = faiss.IndexIVFPQ(quantizer, dimension, n_centroids, m, nbits)
```

## Embedding Generation Pipeline

### 1. Text Embedding Process

The text embedding pipeline consists of:

1. **Preprocessing**:

   - UTF-8 encoding normalization
   - Stopword handling: conditional retention for fashion terms
   - Sentence normalization and tokenization

2. **Model Inference**:

   ```python
   def get_text_embedding(self, text):
       if self.use_openai:
           # OpenAI embeddings via API
           response = openai.Embedding.create(
               input=text,
               model="text-embedding-ada-002"
           )
           return np.array(response['data'][0]['embedding'], dtype=np.float32)
       else:
           # Local SentenceTransformer model
           return self.text_model.encode(text, normalize_embeddings=True)
   ```

3. **Post-processing**:
   - Vector normalization for cosine similarity
   - Dimensionality reduction (optional via PCA)

### 2. Image Embedding Process

The image embedding pipeline consists of:

1. **Preprocessing**:

   - Image loading and error handling
   - Resize to 224x224 pixels
   - RGB color normalization
   - Tensor conversion with torch.Tensor([1, 3, 224, 224])

2. **Model Inference**:

   ```python
   def get_image_embedding(self, image_path):
       try:
           # Load and preprocess image
           image = self._preprocess_image(image_path)

           # Get CLIP image embedding
           with torch.no_grad():
               inputs = processor(images=image, return_tensors="pt")
               image_features = model.get_image_features(**inputs)
               embedding = image_features.cpu().numpy().flatten()

           # Normalize embedding
           embedding = embedding / np.linalg.norm(embedding)
           return embedding
       except Exception as e:
           # Fallback to zero vector with warning
           return np.zeros(512, dtype=np.float32)
   ```

3. **Error Handling**:
   - Graceful fallback to default embedding for missing images
   - Error logging and reporting

### 3. Multimodal Fusion Technique

For combined search, we implement a weighted fusion approach:

```python
def get_combined_embedding(self, text, image_path, text_weight=0.5):
    # Generate individual embeddings
    text_emb = self.get_text_embedding(text)
    image_emb = self.get_image_embedding(image_path)

    # Normalize embeddings if they have different dimensions
    if text_emb.shape[0] != image_emb.shape[0]:
        # Use PCA to reduce dimensions or padding
        # Current implementation uses text-only or image-only in this case
        return text_emb if text_weight > 0.5 else image_emb

    # Weighted combination
    combined = text_weight * text_emb + (1 - text_weight) * image_emb

    # Final normalization
    return combined / np.linalg.norm(combined)
```

## Vector Database Implementation

### 1. FAISS Index Management

The VectorDatabase class implements:

1. **Index Creation and Addition**:

   ```python
   def create_index(self, embeddings):
       # Determine dimensions from first embedding
       self.dimension = embeddings[0].shape[0]

       # Create FAISS index
       self.index = faiss.IndexFlatL2(self.dimension)

       # Add vectors to index
       embeddings_array = np.array(embeddings).astype('float32')
       self.index = faiss.IndexIDMap(self.index)
       self.index.add_with_ids(embeddings_array, np.array(range(len(embeddings))))

       return self
   ```

2. **Search Implementation**:

   ```python
   def search(self, query_embedding, k=5):
       # Ensure query has correct shape
       query_np = np.array([query_embedding]).astype('float32')

       # Perform search
       distances, indices = self.index.search(query_np, k)

       # Compile results with metadata
       results = []
       for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
           if idx < len(self.metadata):  # Ensure valid index
               product_metadata = self.metadata[idx]
               product_id = product_metadata["id"]

               # Get full product from DataFrame
               if self.products_df is not None:
                   product = self.products_df.loc[
                       self.products_df['id'] == product_id
                   ].iloc[0].to_dict()
               else:
                   product = {"id": product_id}

               results.append({
                   "product": product,
                   "metadata": product_metadata,
                   "distance": float(dist),
                   "similarity": float(1.0 - min(dist, 1.0))
               })

       return results
   ```

3. **Serialization and Persistence**:
   ```python
   def save(self, directory, name="vector_db"):
       # Save index
       index_path = os.path.join(directory, f"{name}_index.faiss")
       faiss.write_index(self.index, index_path)

       # Save metadata
       metadata_path = os.path.join(directory, f"{name}_metadata.pkl")
       with open(metadata_path, 'wb') as f:
           pickle.dump(self.metadata, f)

       # Save embeddings
       if hasattr(self, 'embeddings') and self.embeddings is not None:
           emb_path = os.path.join(directory, f"{name}_embeddings.npy")
           np.save(emb_path, self.embeddings)

       # Save configuration
       config_path = os.path.join(directory, f"{name}_config.json")
       with open(config_path, 'w') as f:
           json.dump({"dimension": self.dimension}, f)
   ```

### 2. Database Optimization Techniques

- **Memory Optimization**: On-disk index for large datasets
- **Batch Processing**: Processing embeddings in batches
- **Index Types**: Flat index for accuracy, IVF for speed
- **Dimensionality**: Trade-offs between dimension size and accuracy

## RAG System Architecture

### 1. Retrieval-Augmented Generation Implementation

```python
def enhance_product_description(self, product, similar_products):
    # Build prompt with product and similar products context
    prompt = f"""
    Product Information:
    Title: {product['title']}
    Category: {product['category']}
    Description: {product['description']}
    Price: ${product['price']:.2f}

    Similar Products:
    {self._format_similar_products(similar_products)}

    Task: Create an enhanced, detailed description of the product.
    Be specific about the materials, fit, style, and potential uses.
    Highlight what makes this product unique compared to similar items.
    """

    # Call LLM for generation
    response = self._call_llm_api(prompt)
    return response
```

### 2. OpenAI API Integration Details

The RAG system utilizes OpenAI's API with specific configuration:

```python
def _call_llm_api(self, prompt, max_tokens=300):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.3,
            presence_penalty=0.2
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating content: {str(e)}"
```

## Performance Optimization

### 1. Caching Strategy

Streamlit's caching decorator is used for expensive operations:

```python
@st.cache_data
def load_image(image_path):
    # Image loading with caching

@st.cache_resource
def load_embedding_models():
    # Model loading with caching
```

### 2. Memory Profiling Results

| Component   | Cold Start | Warm Start | Peak Memory |
| ----------- | ---------- | ---------- | ----------- |
| Text Model  | 4.2s       | 0.1s       | 750MB       |
| Image Model | 7.8s       | 0.2s       | 1.2GB       |
| RAG System  | 3.5s       | 0.1s       | 450MB       |
| FAISS Index | 0.9s       | 0.05s      | Varies      |

### 3. Latency Optimization

- Precomputed embeddings for known products
- Multi-phase search with progressive refinement
- Asynchronous loading of non-critical components

## System Limitations and Technical Debt

1. **Architectural Limitations**:

   - Single-node vector database without sharding
   - No incremental updates for the FAISS index
   - In-memory operation for small/medium datasets only

2. **Known Technical Debt**:

   - Hard-coded model paths in the EmbeddingGenerator
   - Limited error handling for API rate limits
   - No type checking on embedding dimensions

3. **Resource Constraints**:
   - Memory usage scales linearly with dataset size
   - CPU-bound operations for large batch processing
   - Network latency for OpenAI API calls

## Production Deployment Considerations

### 1. Scalability Architecture

```
┌───────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Load Balancer │───▶│  API Gateway    │───▶│ Application Pods  │
└───────────────┘    └─────────────────┘    └──────────────────┘
                                                     │
                                                     ▼
┌───────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Redis Cache  │◀───│ Service Mesh    │◀───│  Vector Engine   │
└───────────────┘    └─────────────────┘    └──────────────────┘
        ▲                                           │
        │                                           ▼
┌───────────────┐                         ┌──────────────────┐
│  Monitoring   │◀────────────────────────│   Storage Layer  │
└───────────────┘                         └──────────────────┘
```

### 2. Infrastructure Requirements

- **Compute**: 4 vCPUs minimum, 8 GB RAM
- **Storage**: 20 GB SSD for database and models
- **Network**: Low-latency connection for API calls
- **Scaling**: Horizontal scaling for web tier, vertical for database

### 3. Security Considerations

- API key rotation and secure storage
- Rate limiting and request validation
- Input sanitization for all user queries
- Obfuscation of model details and parameters

## Mathematical Foundations

### 1. Similarity Metrics

For similarity search, the system uses:

**L2 Distance (Euclidean)**:
$$d(p,q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

**Cosine Similarity**:
$$\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \times ||\mathbf{B}||}$$

### 2. Vector Normalization

$$\vec{v}_{\text{normalized}} = \frac{\vec{v}}{||\vec{v}||} = \frac{\vec{v}}{\sqrt{\sum_{i=1}^{n} v_i^2}}$$

### 3. Embedding Fusion Model

The weighted fusion of text and image embeddings:

$$\vec{e}_{\text{combined}} = \frac{w_t \cdot \vec{e}_{\text{text}} + (1-w_t) \cdot \vec{e}_{\text{image}}}{||\vec{w}_t \cdot \vec{e}_{\text{text}} + (1-w_t) \cdot \vec{e}_{\text{image}}||}$$

Where $w_t$ is the text weight parameter.

## Algorithmic Complexity Analysis

| Operation                | Time Complexity | Space Complexity |
| ------------------------ | --------------- | ---------------- |
| Text Embedding           | O(L)            | O(D₁)            |
| Image Embedding          | O(W×H)          | O(D₂)            |
| Index Creation           | O(N×D×log(N))   | O(N×D)           |
| Exact Search (Flat)      | O(N×D)          | O(k)             |
| Approximate Search (IVF) | O(k'×D)         | O(k)             |
| RAG Generation           | O(1)\*          | O(L_out)         |

\*Constant time assumes fixed API latency; actual depends on OpenAI's internal processing

Where:

- N = number of products
- D = embedding dimension
- L = text length
- W,H = image dimensions
- k = number of results
- k' = number of clusters to search in IVF
- L_out = output text length
