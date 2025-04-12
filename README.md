# AI Product Recommendation System

A multimodal AI agent built with vector search, similarity search, embeddings, and RAG to power an e-commerce recommendation system.

## Project Overview

This project demonstrates a complete AI product recommendation system with the following components:

1. **Data Preparation**: Collection of a small fashion product dataset with titles, descriptions, and images
2. **Embedding Generation**: Text embeddings created with SentenceTransformers/OpenAI and image embeddings with CLIP
3. **Vector Database**: Storage of embeddings in FAISS (Facebook AI Similarity Search)
4. **Similarity Search & RAG**: Implementation of similarity search for recommendations and RAG for enhanced content
5. **Mobile App Prototype**: A Streamlit interface with multimodal search capabilities

## Features

- Text-based search for products using natural language
- Image-based search for visually similar products
- Combined text+image search for more accurate recommendations
- RAG-enhanced product descriptions, comparisons, and styling tips
- Interactive UI for browsing and discovering products

## Installation

1. Clone this repository

```bash
git clone https://github.com/yourusername/ai-product-recommendations.git
cd ai-product-recommendations
```

2. Install the required dependencies

```bash
pip install -r requirements.txt
```

3. (Optional) Set up OpenAI API key for enhanced RAG features
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

### Running the Streamlit App

```bash
cd app
streamlit run app.py
```

This will start the Streamlit app on `http://localhost:8501` where you can:

- Search by text description
- Upload images to find similar products
- Browse the product catalog
- View AI-enhanced product descriptions and comparisons

## Project Structure

```
├── app/
│   ├── components/         # UI components
│   ├── utils/              # Utility functions
│   │   ├── data_loader.py          # Dataset creation
│   │   ├── embedding_generator.py  # Text and image embeddings
│   │   ├── vector_database.py      # FAISS database implementation
│   │   ├── rag_generator.py        # RAG functionality
│   │   └── build_database.py       # Database initialization
│   └── app.py              # Main Streamlit application
├── data/
│   ├── images/             # Product images
│   ├── vector_db/          # Stored vector databases
│   ├── products.csv        # Product metadata
│   └── products.json       # Product data in JSON format
├── .env                    # Environment variables (create this)
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Technologies

- **Python**: Core programming language
- **Streamlit**: UI framework for the mobile app prototype
- **FAISS**: Vector database for similarity search
- **Sentence-Transformers**: Text embedding model
- **CLIP**: Image embedding model
- **OpenAI API**: RAG functionality for enhanced content
- **Pandas**: Data manipulation
- **PIL/Pillow**: Image processing

