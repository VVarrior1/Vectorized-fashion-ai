import os
import pandas as pd
from .data_loader import create_sample_dataset
from .embedding_generator import EmbeddingGenerator
from .vector_database import VectorDatabase

def build_product_database(use_openai=False, force_rebuild=False):
    """
    Build the product database with embeddings
    
    Args:
        use_openai: Whether to use OpenAI for text embeddings
        force_rebuild: Whether to force rebuild the database even if it exists
        
    Returns:
        Tuple of (products_df, text_db, image_db, combined_db)
    """
    # Define paths and ensure directories exist
    data_dir = "data"
    vector_db_dir = os.path.join(data_dir, "vector_db")
    products_path = os.path.join(data_dir, "products.csv")
    db_version_file = os.path.join(data_dir, "db_version.txt")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
    os.makedirs(vector_db_dir, exist_ok=True)
    
    # Check database version to determine if rebuild is needed
    current_version = "2.0"  # Increment this when making significant data changes
    stored_version = None
    
    if os.path.exists(db_version_file):
        with open(db_version_file, 'r') as f:
            stored_version = f.read().strip()
    
    # Force rebuild if version changed
    if stored_version != current_version:
        print(f"Database version changed ({stored_version} -> {current_version}). Forcing rebuild.")
        force_rebuild = True
    
    # Check if database already exists
    text_db_exists = (os.path.exists(os.path.join(vector_db_dir, "text_db_index.faiss")) or 
                      os.path.exists(os.path.join(vector_db_dir, "text_db_embeddings.npy")))
    image_db_exists = (os.path.exists(os.path.join(vector_db_dir, "image_db_index.faiss")) or 
                       os.path.exists(os.path.join(vector_db_dir, "image_db_embeddings.npy")))
    combined_db_exists = (os.path.exists(os.path.join(vector_db_dir, "combined_db_index.faiss")) or 
                          os.path.exists(os.path.join(vector_db_dir, "combined_db_embeddings.npy")))
    
    # Load or create product dataset
    if os.path.exists(products_path) and not force_rebuild:
        print("Loading existing product dataset...")
        products_df = pd.read_csv(products_path)
    else:
        print("Creating sample product dataset...")
        products_df = create_sample_dataset(output_dir=data_dir)
    
    # If databases exist and not forcing rebuild, load them
    if text_db_exists and image_db_exists and combined_db_exists and not force_rebuild:
        print("Loading existing vector databases...")
        try:
            text_db = VectorDatabase.load(vector_db_dir, "text_db")
            image_db = VectorDatabase.load(vector_db_dir, "image_db")
            combined_db = VectorDatabase.load(vector_db_dir, "combined_db")
            
            # Load product data
            text_db.load_product_data(products_df)
            image_db.load_product_data(products_df)
            combined_db.load_product_data(products_df)
            
            return products_df, text_db, image_db, combined_db
        except Exception as e:
            print(f"Error loading existing databases: {str(e)}")
            print("Rebuilding databases...")
    
    # Create embeddings
    print("Generating embeddings...")
    embedding_generator = EmbeddingGenerator(use_openai=use_openai)
    
    # Create text embeddings
    product_texts = [
        f"{row['title']}. {row['description']} Category: {row['category']}"
        for _, row in products_df.iterrows()
    ]
    text_embeddings = embedding_generator.batch_get_text_embeddings(product_texts)
    
    # Create image embeddings
    image_paths = [
        os.path.join(data_dir, "images", f"product_{row['id']}.jpg")
        for _, row in products_df.iterrows()
    ]
    image_embeddings = embedding_generator.batch_get_image_embeddings(image_paths)
    
    # Create combined embeddings (simple implementation - in real app might be more sophisticated)
    combined_embeddings = []
    for i, row in products_df.iterrows():
        text = f"{row['title']}. {row['description']} Category: {row['category']}"
        image_path = os.path.join(data_dir, "images", f"product_{row['id']}.jpg")
        combined_embedding = embedding_generator.get_combined_embedding(text, image_path)
        combined_embeddings.append(combined_embedding)
    
    # Create metadata list
    metadata_list = [
        {
            "id": row["id"],
            "title": row["title"],
            "category": row["category"]
        }
        for _, row in products_df.iterrows()
    ]
    
    # Create and save vector databases
    os.makedirs(vector_db_dir, exist_ok=True)
    
    # Text database
    text_db = VectorDatabase().create_index(text_embeddings)
    text_db.metadata = metadata_list
    text_db.load_product_data(products_df)
    text_db.save(vector_db_dir, "text_db")
    
    # Image database
    image_db = VectorDatabase().create_index(image_embeddings)
    image_db.metadata = metadata_list
    image_db.load_product_data(products_df)
    image_db.save(vector_db_dir, "image_db")
    
    # Combined database
    combined_db = VectorDatabase().create_index(combined_embeddings)
    combined_db.metadata = metadata_list
    combined_db.load_product_data(products_df)
    combined_db.save(vector_db_dir, "combined_db")
    
    # Update version file
    with open(db_version_file, 'w') as f:
        f.write(current_version)
    
    print("Vector databases created and saved successfully.")
    return products_df, text_db, image_db, combined_db


if __name__ == "__main__":
    # Example usage
    build_product_database(use_openai=False, force_rebuild=True) 