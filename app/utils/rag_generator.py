import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGGenerator:
    def __init__(self):
        """Initialize the RAG generator with OpenAI API"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for RAG")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"  # Can be changed to gpt-4 for better results
    
    def enhance_product_description(self, product, similar_products=None):
        """
        Generate an enhanced product description using RAG
        
        Args:
            product: Dictionary containing product information
            similar_products: List of similar products for context
            
        Returns:
            Enhanced product description
        """
        # Build the prompt with product information
        prompt = f"Generate an enhanced, engaging product description for the following fashion item:\n\n"
        prompt += f"Product: {product.get('title', 'Unknown Product')}\n"
        prompt += f"Category: {product.get('category', 'Unknown Category')}\n"
        prompt += f"Original Description: {product.get('description', 'No description available')}\n"
        prompt += f"Price: ${product.get('price', 0.0)}\n\n"
        
        # Add context from similar products if available
        if similar_products and len(similar_products) > 0:
            prompt += "Here are some similar products for context:\n"
            for i, similar in enumerate(similar_products[:3]):  # Limit to 3 similar products
                p = similar.get("product", {})
                prompt += f"{i+1}. {p.get('title', 'Unknown')}: {p.get('description', 'No description')}\n"
        
        prompt += "\nPlease generate an improved, detailed, and engaging product description that highlights its features, benefits, and potential uses. Make it appealing to customers while being accurate to the product category."
        
        # Generate the enhanced description
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional e-commerce product copywriter with expertise in fashion and retail. Your job is to create compelling, accurate, and engaging product descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        
        return response.choices[0].message.content
    
    def generate_product_comparison(self, main_product, comparison_products):
        """
        Generate a comparison between products using RAG
        
        Args:
            main_product: Dictionary containing the main product information
            comparison_products: List of products to compare with
            
        Returns:
            Product comparison text
        """
        # Build the prompt with products to compare
        prompt = f"Generate a comparison between these fashion products:\n\n"
        
        # Main product
        prompt += f"Main Product: {main_product.get('title', 'Unknown Product')}\n"
        prompt += f"Category: {main_product.get('category', 'Unknown Category')}\n"
        prompt += f"Description: {main_product.get('description', 'No description available')}\n"
        prompt += f"Price: ${main_product.get('price', 0.0)}\n\n"
        
        # Comparison products
        prompt += "Comparison Products:\n"
        for i, product in enumerate(comparison_products[:3]):  # Limit to 3 comparison products
            p = product.get("product", {})
            prompt += f"{i+1}. {p.get('title', 'Unknown')}\n"
            prompt += f"   Category: {p.get('category', 'Unknown')}\n"
            prompt += f"   Description: {p.get('description', 'No description')}\n"
            prompt += f"   Price: ${p.get('price', 0.0)}\n\n"
        
        prompt += "Please generate a balanced comparison between the main product and the comparison products. Highlight the unique features, pros and cons of each, and suggest who might prefer each product based on style, budget, and features."
        
        # Generate the comparison
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional retail analyst with expertise in fashion and consumer products. Your job is to create objective, informative, and balanced product comparisons."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7,
        )
        
        return response.choices[0].message.content
    
    def generate_styling_suggestions(self, product, similar_products=None):
        """
        Generate styling suggestions for a product using RAG
        
        Args:
            product: Dictionary containing product information
            similar_products: List of similar products for context
            
        Returns:
            Styling suggestions text
        """
        # Build the prompt
        prompt = f"Generate styling suggestions for the following fashion item:\n\n"
        prompt += f"Product: {product.get('title', 'Unknown Product')}\n"
        prompt += f"Category: {product.get('category', 'Unknown Category')}\n"
        prompt += f"Description: {product.get('description', 'No description available')}\n\n"
        
        # Add context from similar products if available
        if similar_products and len(similar_products) > 0:
            prompt += "Here are some similar or complementary products that could be styled with it:\n"
            for i, similar in enumerate(similar_products[:5]):  # Include more products for styling ideas
                p = similar.get("product", {})
                prompt += f"{i+1}. {p.get('title', 'Unknown')} ({p.get('category', 'Unknown Category')})\n"
        
        prompt += "\nPlease generate three different styling suggestions for this product, including occasions where it could be worn, complementary items to pair with it, and styling tips. Make the suggestions practical, fashionable, and aligned with current trends."
        
        # Generate the styling suggestions
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional fashion stylist with expertise in creating versatile and trendy outfits. Your job is to provide practical and fashionable styling suggestions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.8,
        )
        
        return response.choices[0].message.content 