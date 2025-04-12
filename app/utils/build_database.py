import os
import pandas as pd
from .data_loader import create_sample_dataset
from .embedding_generator import EmbeddingGenerator
from .vector_database import VectorDatabase

def build_product_database(use_openai=False, force_rebuild=False):
    """Build the product database with embeddings"""
    data_dir = "data"
    vector_db_dir = os.path.join(data_dir, "vector_db")
    products_path = os.path.join(data_dir, "products.csv")
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
    os.makedirs(vector_db_dir, exist_ok=True)
    
    # Check if database exists
    dbs_exist = all(
        os.path.exists(os.path.join(vector_db_dir, f"{db}_index.faiss"))
        for db in ["text_db", "image_db", "combined_db"]
    )
    
    # Load or create product dataset
    if os.path.exists(products_path) and not force_rebuild:
        products_df = pd.read_csv(products_path)
    else:
        products_df = create_sample_dataset(output_dir=data_dir)
    
    # Load existing databases if available
    if dbs_exist and not force_rebuild:
        try:
            text_db = VectorDatabase.load(vector_db_dir, "text_db")
            image_db = VectorDatabase.load(vector_db_dir, "image_db")
            combined_db = VectorDatabase.load(vector_db_dir, "combined_db")
            
            for db in [text_db, image_db, combined_db]:
                db.load_product_data(products_df)
            
            return products_df, text_db, image_db, combined_db
        except Exception as e:
            print(f"Error loading databases: {str(e)}. Rebuilding...")
    
    # Generate embeddings
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
    
    # Create combined embeddings
    combined_embeddings = [
        embedding_generator.get_combined_embedding(
            f"{row['title']}. {row['description']} Category: {row['category']}", 
            os.path.join(data_dir, "images", f"product_{row['id']}.jpg")
        )
        for _, row in products_df.iterrows()
    ]
    
    # Create metadata
    metadata_list = [
        {"id": row["id"], "title": row["title"], "category": row["category"]}
        for _, row in products_df.iterrows()
    ]
    
    # Create and save databases
    databases = {
        "text_db": text_embeddings,
        "image_db": image_embeddings,
        "combined_db": combined_embeddings
    }
    
    result_dbs = {}
    for name, embeddings in databases.items():
        db = VectorDatabase().create_index(embeddings)
        db.metadata = metadata_list
        db.load_product_data(products_df)
        db.save(vector_db_dir, name)
        result_dbs[name] = db
    
    return (products_df, result_dbs["text_db"], 
            result_dbs["image_db"], result_dbs["combined_db"])


if __name__ == "__main__":
    # Example usage
    build_product_database(use_openai=False, force_rebuild=True) 