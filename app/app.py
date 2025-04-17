import os
import streamlit as st
import pandas as pd
from PIL import Image
import io
import time
import re
from utils.embedding_generator import EmbeddingGenerator
from utils.vector_database import VectorDatabase
from utils.rag_generator import RAGGenerator
from utils.build_database import build_product_database
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)
os.makedirs(os.path.join("data", "images"), exist_ok=True)

# Function to load image from path or URL
@st.cache_data
def load_image(image_path):
    try:
        # Check if path exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            return None
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

# Function to get a human-readable category name
def format_category(category):
    if not category:
        return "Unknown"
    
    # Split by underscore
    if "_" in category:
        main, sub = category.split("_", 1)
        return f"{main.title()} - {sub.title()}"
    
    # Otherwise just capitalize
    return category.title()

# Initialize session state for storing recommendations across reruns
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'current_product' not in st.session_state:
    st.session_state.current_product = None
if 'tab' not in st.session_state:
    st.session_state.tab = "text_search"

# Main function to run the app
def main():
    # Ensure session state is initialized
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'current_product' not in st.session_state:
        st.session_state.current_product = None
    if 'tab' not in st.session_state:
        st.session_state.tab = "text_search"
    
    st.title("👗 Fashion Product Recommender")
    st.markdown("Discover fashion items with AI-powered recommendations")
    
    # Initialize databases
    with st.spinner("Loading product database..."):
        try:
            use_openai = os.getenv("OPENAI_API_KEY") is not None
            products_df, text_db, image_db, combined_db = build_product_database(use_openai=use_openai)
        except Exception as e:
            st.error(f"Error loading database: {str(e)}")
            if os.getenv("OPENAI_API_KEY") is None:
                st.warning("OPENAI_API_KEY is not set. Using SentenceTransformer for text embeddings.")
            
            # Try without OpenAI
            products_df, text_db, image_db, combined_db = build_product_database(use_openai=False)
    
    # Create tabs for different search methods
    tab1, tab2, tab3 = st.tabs(["Text Search", "Image Upload", "Browse Products"])
    
    # Tab 1: Text Search
    with tab1:
        st.header("Search by Text")
        search_query = st.text_input("Enter a description of what you're looking for:")
        
        # Filter options
        st.write("Filter options (optional):")
        col1, col2 = st.columns(2)
        
        # Extract main categories for filtering
        main_categories = sorted(list(set([
            cat.split('_')[0] if '_' in cat else cat 
            for cat in products_df['category'].unique()
        ])))
        
        with col1:
            selected_main_category = st.selectbox(
                "Main Category:", 
                ["All"] + main_categories
            )
        
        with col2:
            # Price range
            price_range = st.slider(
                "Price Range ($):", 
                min_value=0, 
                max_value=int(products_df['price'].max() + 20),
                value=(0, int(products_df['price'].max())),
                step=10
            )
        
        search_button = st.button("Search", key="text_search_button")
        
        if search_button and search_query:
            with st.spinner("Searching for products..."):
                # Generate text embedding for the search query
                embedding_generator = EmbeddingGenerator(use_openai=use_openai)
                query_embedding = embedding_generator.get_text_embedding(search_query)
                
                # Search the text database - get more results than needed for filtering
                initial_results = text_db.search(query_embedding, k=20)
                
                # Apply filters
                filtered_results = []
                for result in initial_results:
                    product = result["product"]
                    
                    # Category filter
                    if selected_main_category != "All":
                        if '_' in product['category']:
                            main_cat = product['category'].split('_')[0]
                        else:
                            main_cat = product['category']
                            
                        if main_cat != selected_main_category:
                            continue
                    
                    # Price filter
                    if not (price_range[0] <= product['price'] <= price_range[1]):
                        continue
                    
                    filtered_results.append(result)
                
                # Limit to top 5 filtered results
                filtered_results = filtered_results[:5]
                
                if filtered_results:
                    st.session_state.recommendations = filtered_results
                    st.session_state.tab = "text_search"
                    
                    # Display the first result detail
                    st.session_state.current_product = filtered_results[0]["product"]
                else:
                    st.warning("No products match your search criteria. Try adjusting your filters.")
    
    # Tab 2: Image Upload
    with tab2:
        st.header("Search by Image")
        uploaded_file = st.file_uploader("Upload an image of a fashion item:", type=["jpg", "jpeg", "png"])
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            selected_main_category = st.selectbox(
                "Filter by Category:", 
                ["All"] + main_categories,
                key="image_category_filter"
            )
            
        with col2:
            search_method = st.selectbox(
                "Search Method:",
                ["Visual Similarity", "Combined (Visual + Text)"]
            )
        
        if uploaded_file:
            # Save the uploaded image temporarily
            image = Image.open(uploaded_file).convert("RGB")
            temp_image_path = os.path.join("data", "temp_upload.jpg")
            os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
            image.save(temp_image_path)
            
            # Show the uploaded image
            st.image(image, caption="Uploaded Image", width=200)
            
            # Search button
            search_image_button = st.button("Find Similar Items", key="image_search_button")
            
            if search_image_button:
                with st.spinner("Analyzing image and finding similar products..."):
                    # Generate image embedding
                    embedding_generator = EmbeddingGenerator(use_openai=use_openai)
                    query_embedding = embedding_generator.get_image_embedding(temp_image_path)
                    
                    # Choose database based on search method
                    if search_method == "Visual Similarity":
                        search_db = image_db
                    else:  # Combined
                        search_db = combined_db
                        # For combined search, we need both text and image
                        # Since we don't have text from the user for the uploaded image,
                        # we'll use a generic description
                        generic_desc = "Fashion item, clothing, apparel"
                        text_embedding = embedding_generator.get_text_embedding(generic_desc)
                        # Get a combined embedding
                        query_embedding = embedding_generator.get_combined_embedding(
                            generic_desc, temp_image_path, text_weight=0.3
                        )
                    
                    # Search the database - get more results for filtering
                    initial_results = search_db.search(query_embedding, k=20)
                    
                    # Apply category filter
                    filtered_results = []
                    for result in initial_results:
                        product = result["product"]
                        
                        # Category filter
                        if selected_main_category != "All":
                            if '_' in product['category']:
                                main_cat = product['category'].split('_')[0]
                            else:
                                main_cat = product['category']
                                
                            if main_cat != selected_main_category:
                                continue
                        
                        filtered_results.append(result)
                    
                    # Limit to top 5 filtered results
                    filtered_results = filtered_results[:5]
                    
                    if filtered_results:
                        st.session_state.recommendations = filtered_results
                        st.session_state.tab = "image_search"
                        
                        # Display the first result detail
                        st.session_state.current_product = filtered_results[0]["product"]
                    else:
                        st.warning("No visually similar products found in the selected category.")
    
    # Tab 3: Browse Products
    with tab3:
        st.header("Browse All Products")
        
        # Get all unique main categories and subcategories
        categories = products_df["category"].unique().tolist()
        main_categories = sorted(list(set([
            cat.split('_')[0] if '_' in cat else cat 
            for cat in categories
        ])))
        
        # Main category selection
        selected_main = st.selectbox(
            "Main Category:", 
            ["All Categories"] + main_categories
        )
        
        # Subcategory selection based on main category
        if selected_main != "All Categories":
            subcategories = sorted(list(set([
                cat.split('_')[1] if '_' in cat and cat.startswith(selected_main) else None
                for cat in categories
            ])))
            subcategories = [sub for sub in subcategories if sub is not None]
            
            selected_sub = st.selectbox(
                "Subcategory:",
                ["All"] + subcategories
            )
            
            # Filter based on selection
            if selected_sub == "All":
                filtered_df = products_df[products_df["category"].str.startswith(selected_main + "_")]
            else:
                filtered_df = products_df[products_df["category"] == f"{selected_main}_{selected_sub}"]
        else:
            filtered_df = products_df
        
        # Create a grid layout for products
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
        
        for i, (_, product) in enumerate(filtered_df.iterrows()):
            col = columns[i % 3]
            with col:
                image_path = os.path.join("data", "images", f"product_{product['id']}.jpg")
                product_image = load_image(image_path)
                
                if product_image:
                    st.image(product_image, caption=product["title"], width=150)
                
                # Display category nicely
                formatted_category = format_category(product["category"])
                st.caption(f"{formatted_category} - ${product['price']:.2f}")
                
                # View Details button
                if st.button(f"View Details", key=f"product_{product['id']}"):
                    st.session_state.current_product = product.to_dict()
                    
                    # Find similar products
                    embedding_generator = EmbeddingGenerator(use_openai=use_openai)
                    product_text = f"{product['title']}. {product['description']} Category: {product['category']}"
                    query_embedding = embedding_generator.get_text_embedding(product_text)
                    
                    results = text_db.search(query_embedding, k=6)  # k=6 to include the product itself
                    st.session_state.recommendations = [r for r in results if r["metadata"]["id"] != product["id"]]
                    st.session_state.tab = "browse"
    
    # Display recommendations and product details
    if st.session_state.current_product:
        st.markdown("---")
        
        # Layout with columns
        col1, col2 = st.columns([1, 2])
        
        # Product image and basic info in column 1
        with col1:
            product = st.session_state.current_product
            st.subheader(product["title"])
            
            # Display product image
            image_path = os.path.join("data", "images", f"product_{product['id']}.jpg")
            product_image = load_image(image_path)
            if product_image:
                st.image(product_image, width=250)
            
            # Display category nicely
            formatted_category = format_category(product["category"])
            st.write(f"**Category:** {formatted_category}")
            st.write(f"**Price:** ${product['price']:.2f}")
            
            # Original description
            st.write("**Description:**")
            st.write(product["description"])
        
        # Enhanced details in column 2
        with col2:
            # Create RAG generator if OpenAI API key is available
            if os.getenv("OPENAI_API_KEY"):
                rag_generator = RAGGenerator()
                
                # Create tabs for different enhanced content
                enhanced_tab1, enhanced_tab2, enhanced_tab3 = st.tabs(["Enhanced Description", "Styling Tips", "Product Comparison"])
                
                with enhanced_tab1:
                    with st.spinner("Generating enhanced description..."):
                        enhanced_description = rag_generator.enhance_product_description(
                            product, 
                            st.session_state.recommendations[:3]
                        )
                        st.write(enhanced_description)
                
                with enhanced_tab2:
                    with st.spinner("Generating styling suggestions..."):
                        styling_tips = rag_generator.generate_styling_suggestions(
                            product, 
                            st.session_state.recommendations
                        )
                        st.write(styling_tips)
                
                with enhanced_tab3:
                    with st.spinner("Generating product comparison..."):
                        if len(st.session_state.recommendations) > 0:
                            comparison = rag_generator.generate_product_comparison(
                                product, 
                                st.session_state.recommendations[:3]
                            )
                            st.write(comparison)
                        else:
                            st.info("No similar products found for comparison.")
            else:
                st.warning("OpenAI API key not found. Enhanced descriptions, styling tips, and comparisons are not available.")
        
        # Display similar products
        st.markdown("---")
        st.subheader("Similar Products You Might Like")
        
        if st.session_state.recommendations and len(st.session_state.recommendations) > 0:
            # Create a grid layout for similar products
            sim_cols = st.columns(min(4, len(st.session_state.recommendations)))
            
            for i, result in enumerate(st.session_state.recommendations[:4]):  # Display up to 4 similar products
                with sim_cols[i]:
                    similar_product = result["product"]
                    st.write(f"**{similar_product['title']}**")
                    
                    # Display product image
                    sim_image_path = os.path.join("data", "images", f"product_{similar_product['id']}.jpg")
                    sim_product_image = load_image(sim_image_path)
                    if sim_product_image:
                        st.image(sim_product_image, width=150)
                    
                    # Display category nicely
                    formatted_category = format_category(similar_product["category"])
                    st.caption(f"{formatted_category}")
                    st.write(f"Price: ${similar_product['price']:.2f}")
                    st.write(f"Similarity: {100 - result['distance']*10:.1f}%")
                    
                    # View product button
                    if st.button(f"View Details", key=f"sim_product_{similar_product['id']}"):
                        st.session_state.current_product = similar_product
                        
                        # Find similar products to this one
                        embedding_generator = EmbeddingGenerator(use_openai=use_openai)
                        product_text = f"{similar_product['title']}. {similar_product['description']} Category: {similar_product['category']}"
                        query_embedding = embedding_generator.get_text_embedding(product_text)
                        
                        results = text_db.search(query_embedding, k=6)
                        st.session_state.recommendations = [r for r in results if r["metadata"]["id"] != similar_product["id"]]
                        
                        # Rerun to update the display
                        st.rerun()
        else:
            st.info("No similar products found.")

# Create a .env file if it doesn't exist
if not os.path.exists(".env"):
    with open(".env", "w") as f:
        f.write("# Add your OpenAI API key here\n")
        f.write("# OPENAI_API_KEY=your_api_key_here\n")

# This block will only run when app.py is run directly, not when imported
if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="Fashion Recommender",
        page_icon="👔",
        layout="wide"
    )
    main() 