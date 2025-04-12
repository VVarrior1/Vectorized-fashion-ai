import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import json

def download_image(url, save_path):
    """Download an image from URL and save it to specified path"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(save_path)
            return True
        return False
    except Exception as e:
        print(f"Error downloading image {url}: {str(e)}")
        return False

def create_sample_dataset(output_dir="data", n_samples=40):
    """Create a sample fashion product dataset using a simple predefined list"""
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Sample product data - Expanded with more products and improved categorization
    products = [
        # TOPS
        {
            "id": 1,
            "title": "Blue Denim Jacket",
            "description": "Classic blue denim jacket with button closure. Perfect for casual outings.",
            "category": "tops_outerwear",
            "price": 89.99,
            "image_url": "https://images.unsplash.com/photo-1576871337622-98d48d1cf531?w=600&auto=format"
        },
        {
            "id": 2,
            "title": "White Cotton T-shirt",
            "description": "Soft and breathable cotton t-shirt. Basic essential for any wardrobe.",
            "category": "tops_tshirts",
            "price": 24.99,
            "image_url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=600&auto=format"
        },
        {
            "id": 3,
            "title": "Red Knit Sweater",
            "description": "Warm knit sweater in vibrant red. Perfect for colder days.",
            "category": "tops_sweaters",
            "price": 69.99,
            "image_url": "https://images.unsplash.com/photo-1584273143981-41c073dfe8f8?w=600&auto=format"
        },
        {
            "id": 4,
            "title": "Striped Polo Shirt",
            "description": "Cotton polo shirt with classic stripes. Casual yet refined look.",
            "category": "tops_tshirts",
            "price": 39.99,
            "image_url": "https://images.unsplash.com/photo-1581655353564-df123a1eb820?w=600&auto=format"
        },
        {
            "id": 5,
            "title": "Blue Formal Shirt",
            "description": "Crisp blue formal shirt with standard collar. Perfect for office wear.",
            "category": "tops_formal",
            "price": 59.99,
            "image_url": "https://images.unsplash.com/photo-1563630423918-b58f07336ac5?w=600&auto=format"
        },
        {
            "id": 6,
            "title": "Green Hoodie",
            "description": "Comfortable cotton hoodie in forest green. Warm and casual.",
            "category": "tops_outerwear",
            "price": 49.99,
            "image_url": "https://images.unsplash.com/photo-1556172732-bcded72dab6b?w=600&auto=format"
        },
        {
            "id": 7,
            "title": "Beige Trench Coat",
            "description": "Classic trench coat in beige. Timeless style and weather-resistant.",
            "category": "tops_outerwear",
            "price": 129.99,
            "image_url": "https://images.unsplash.com/photo-1520975954732-35dd22299614?w=600&auto=format"
        },
        {
            "id": 8,
            "title": "Black Leather Jacket",
            "description": "Sleek black leather jacket with zippered front. Edgy and versatile.",
            "category": "tops_outerwear",
            "price": 179.99,
            "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=600&auto=format"
        },
        {
            "id": 9,
            "title": "Navy Blue Blazer",
            "description": "Tailored navy blue blazer. Perfect for business meetings or formal events.",
            "category": "tops_formal",
            "price": 149.99,
            "image_url": "https://images.unsplash.com/photo-1592878904946-b3cd8ae243d0?w=600&auto=format"
        },
        {
            "id": 10,
            "title": "Graphic Print T-shirt",
            "description": "Cotton t-shirt with artistic graphic print. Stand out from the crowd.",
            "category": "tops_tshirts",
            "price": 32.99,
            "image_url": "https://images.unsplash.com/photo-1503341338985-c0477be52513?w=600&auto=format"
        },
        
        # BOTTOMS
        {
            "id": 11,
            "title": "Black Slim-fit Jeans",
            "description": "Stretch denim slim-fit jeans. Modern cut and comfortable fit.",
            "category": "bottoms_jeans",
            "price": 59.99,
            "image_url": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=600&auto=format"
        },
        {
            "id": 12,
            "title": "Khaki Chino Pants",
            "description": "Classic khaki chino pants. Versatile and perfect for business casual.",
            "category": "bottoms_casual",
            "price": 69.99,
            "image_url": "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=600&auto=format"
        },
        {
            "id": 13,
            "title": "Blue Distressed Jeans",
            "description": "Distressed blue jeans with a relaxed fit. Casual and trendy.",
            "category": "bottoms_jeans",
            "price": 79.99,
            "image_url": "https://images.unsplash.com/photo-1528826007158-4c12fcdf8291?w=600&auto=format"
        },
        {
            "id": 14,
            "title": "Black Dress Pants",
            "description": "Formal black dress pants with a sleek finish. Essential for any formal wardrobe.",
            "category": "bottoms_formal",
            "price": 89.99,
            "image_url": "https://images.unsplash.com/photo-1633966887768-64f9a867bdba?w=600&auto=format"
        },
        {
            "id": 15,
            "title": "Gray Sweatpants",
            "description": "Comfortable gray sweatpants for lounging or workouts. Soft and durable.",
            "category": "bottoms_casual",
            "price": 39.99,
            "image_url": "https://images.unsplash.com/photo-1590739292323-7b1b3d160d97?w=600&auto=format"
        },
        {
            "id": 16,
            "title": "Pleated Black Skirt",
            "description": "Elegant pleated black skirt with an A-line silhouette. Versatile and flattering.",
            "category": "bottoms_skirts",
            "price": 54.99,
            "image_url": "https://images.unsplash.com/photo-1583496661160-fb5886a0aaaa?w=600&auto=format"
        },
        {
            "id": 17,
            "title": "Denim Mini Skirt",
            "description": "Classic denim mini skirt with frayed hem. Casual and stylish.",
            "category": "bottoms_skirts",
            "price": 49.99,
            "image_url": "https://images.unsplash.com/photo-1573211508950-979d142afa90?w=600&auto=format"
        },
        
        # FOOTWEAR
        {
            "id": 18,
            "title": "Black Leather Boots",
            "description": "High-quality leather boots with rubber sole. Water-resistant and durable.",
            "category": "footwear_boots",
            "price": 129.99,
            "image_url": "https://images.unsplash.com/photo-1542840412-ec2c9f6b6030?w=600&auto=format"
        },
        {
            "id": 19,
            "title": "White Sneakers",
            "description": "Clean white leather sneakers. Versatile and minimalist design.",
            "category": "footwear_sneakers",
            "price": 99.99,
            "image_url": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?w=600&auto=format"
        },
        {
            "id": 20,
            "title": "Black Running Shoes",
            "description": "Lightweight running shoes with cushioned sole. Comfortable for long runs.",
            "category": "footwear_athletic",
            "price": 89.99,
            "image_url": "https://images.unsplash.com/photo-1491553895911-0055eca6402d?w=600&auto=format"
        },
        {
            "id": 21,
            "title": "Brown Leather Oxfords",
            "description": "Classic brown leather oxford shoes. Perfect for formal occasions.",
            "category": "footwear_formal",
            "price": 119.99,
            "image_url": "https://images.unsplash.com/photo-1605100804763-247f67b3557e?w=600&auto=format"
        },
        {
            "id": 22,
            "title": "Tan Suede Chelsea Boots",
            "description": "Stylish tan suede Chelsea boots with elastic side panels. Versatile and comfortable.",
            "category": "footwear_boots",
            "price": 149.99,
            "image_url": "https://images.unsplash.com/photo-1638247025967-b4e38f787b76?w=600&auto=format"
        },
        {
            "id": 23,
            "title": "Canvas Slip-on Shoes",
            "description": "Casual canvas slip-on shoes. Easy to wear and comfortable for everyday use.",
            "category": "footwear_casual",
            "price": 49.99,
            "image_url": "https://images.unsplash.com/photo-1573100925118-870b8efc799d?w=600&auto=format"
        },
        
        # DRESSES AND FORMAL WEAR
        {
            "id": 24,
            "title": "Floral Summer Dress",
            "description": "Light floral print dress for summer. Comfortable and stylish.",
            "category": "dresses_casual",
            "price": 79.99,
            "image_url": "https://images.unsplash.com/photo-1623609163859-ca93c959b5b8?w=600&auto=format"
        },
        {
            "id": 25,
            "title": "Gray Formal Suit",
            "description": "Elegant gray suit for formal occasions. Tailored fit and high-quality fabric.",
            "category": "formal_suits",
            "price": 299.99,
            "image_url": "https://images.unsplash.com/photo-1593032465175-481ac7f401a0?w=600&auto=format"
        },
        {
            "id": 26,
            "title": "Black Cocktail Dress",
            "description": "Elegant black cocktail dress with fitted silhouette. Perfect for evening events.",
            "category": "dresses_formal",
            "price": 149.99,
            "image_url": "https://images.unsplash.com/photo-1585487000160-6ebcfceb0d03?w=600&auto=format"
        },
        {
            "id": 27,
            "title": "Navy Blue Tuxedo",
            "description": "Classic navy blue tuxedo with satin lapels. The ultimate formal attire.",
            "category": "formal_suits",
            "price": 349.99,
            "image_url": "https://images.unsplash.com/photo-1598808503746-f34faef0e719?w=600&auto=format"
        },
        {
            "id": 28,
            "title": "Maxi Sundress",
            "description": "Flowing maxi sundress with vibrant pattern. Perfect for beach days or casual outings.",
            "category": "dresses_casual",
            "price": 69.99,
            "image_url": "https://images.unsplash.com/photo-1600073140561-82a93f891827?w=600&auto=format"
        },
        
        # ACCESSORIES
        {
            "id": 29,
            "title": "Brown Leather Wallet",
            "description": "Genuine leather wallet with multiple card slots and coin pocket.",
            "category": "accessories_wallets",
            "price": 49.99,
            "image_url": "https://images.unsplash.com/photo-1559563458-527698bf5295?w=600&auto=format"
        },
        {
            "id": 30,
            "title": "Silver Watch",
            "description": "Elegant silver watch with metal strap. Water-resistant and durable.",
            "category": "accessories_watches",
            "price": 159.99,
            "image_url": "https://images.unsplash.com/photo-1539874754764-5a96559165b0?w=600&auto=format"
        },
        {
            "id": 31,
            "title": "Red Baseball Cap",
            "description": "Adjustable cotton cap with embroidered logo. Perfect for casual outfits.",
            "category": "accessories_hats",
            "price": 29.99,
            "image_url": "https://images.unsplash.com/photo-1521369909029-2afed882baee?w=600&auto=format"
        },
        {
            "id": 32,
            "title": "Navy Blue Backpack",
            "description": "Durable backpack with multiple compartments. Perfect for daily use.",
            "category": "accessories_bags",
            "price": 79.99,
            "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=600&auto=format"
        },
        {
            "id": 33,
            "title": "Patterned Scarf",
            "description": "Soft cotton scarf with geometric pattern. Adds style to any outfit.",
            "category": "accessories_scarves",
            "price": 34.99,
            "image_url": "https://images.unsplash.com/photo-1601106976699-3d7a4699d16c?w=600&auto=format"
        },
        {
            "id": 34,
            "title": "Leather Belt",
            "description": "Classic leather belt with metal buckle. An essential accessory for any wardrobe.",
            "category": "accessories_belts",
            "price": 39.99,
            "image_url": "https://images.unsplash.com/photo-1594225739392-86dd3df75338?w=600&auto=format"
        },
        {
            "id": 35,
            "title": "Black Fedora Hat",
            "description": "Stylish black fedora hat with ribbon trim. Adds sophistication to any outfit.",
            "category": "accessories_hats",
            "price": 59.99,
            "image_url": "https://images.unsplash.com/photo-1514327605112-b887c0e61c0a?w=600&auto=format"
        },
        {
            "id": 36,
            "title": "Leather Tote Bag",
            "description": "Spacious leather tote bag with inner pockets. Perfect for work or shopping.",
            "category": "accessories_bags",
            "price": 119.99,
            "image_url": "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?w=600&auto=format"
        },
        {
            "id": 37,
            "title": "Sunglasses",
            "description": "Classic UV-protective sunglasses with polarized lenses. Stylish and functional.",
            "category": "accessories_eyewear",
            "price": 89.99,
            "image_url": "https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=600&auto=format"
        },
        {
            "id": 38,
            "title": "Gold Necklace",
            "description": "Delicate gold chain necklace with small pendant. Subtle elegance for any occasion.",
            "category": "accessories_jewelry",
            "price": 79.99,
            "image_url": "https://images.unsplash.com/photo-1599643478518-a784e5dc4c8f?w=600&auto=format"
        },
        
        # ELECTRONICS AND OTHERS
        {
            "id": 39,
            "title": "Wireless Headphones",
            "description": "Premium wireless headphones with noise cancellation. Long battery life.",
            "category": "electronics_audio",
            "price": 199.99,
            "image_url": "https://images.unsplash.com/photo-1546435770-a3e426bf472b?w=600&auto=format"
        },
        {
            "id": 40,
            "title": "Smartwatch",
            "description": "Feature-rich smartwatch with fitness tracking and notifications. Water-resistant.",
            "category": "electronics_wearables",
            "price": 249.99,
            "image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600&auto=format"
        }
    ]
    
    # Limit number of products if necessary
    products = products[:n_samples]
    
    # Download images
    for product in products:
        img_path = os.path.join(output_dir, "images", f"product_{product['id']}.jpg")
        success = download_image(product["image_url"], img_path)
        if success:
            product["image_path"] = img_path
        else:
            product["image_path"] = ""
    
    # Create DataFrame
    df = pd.DataFrame(products)
    
    # Save as CSV
    df.to_csv(os.path.join(output_dir, "products.csv"), index=False)
    
    # Save as JSON
    with open(os.path.join(output_dir, "products.json"), 'w') as f:
        json.dump(products, f, indent=2)
    
    print(f"Created sample dataset with {len(products)} products.")
    return df

if __name__ == "__main__":
    create_sample_dataset() 