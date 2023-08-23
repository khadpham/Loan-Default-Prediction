import random
import pandas as pd
import numpy as np
from faker import Faker
import datetime

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Initialize Faker for generating fake data
fake = Faker()

# Define categories, price ranges, and quantities
categories = ["Electronics", "Clothing", "Home & Garden", "Books", "Sports & Outdoors", "Health & Beauty", "Toys & Games", "Office Stationery"]
price_ranges = {
    "Electronics": (5, 1000),  # Updated minimum price for all categories
    "Clothing": (5, 200),      # Updated minimum price for all categories
    "Home & Garden": (5, 500),  # Updated minimum price for all categories
    "Books": (5, 50),           # Updated minimum price for all categories
    "Sports & Outdoors": (5, 300),  # Updated minimum price for all categories
    "Health & Beauty": (5, 150),   # Updated minimum price for all categories
    "Toys & Games": (5, 100),      # Updated minimum price for all categories
    "Office Stationery": (5, 50)   # Updated minimum price for all categories
}
quantities = [1, 1, 1, 2, 2, 3]

# Define real brand names for each category
brand_names = {
    "Electronics": ["Apple", "Samsung", "Sony", "LG", "Microsoft", "Dell", "HP", "Lenovo", "Canon", "Nikon"],
    "Clothing": ["Nike", "Adidas", "H&M", "Zara", "Levi's", "Under Armour", "Gap", "Forever 21"],
    "Home & Garden": ["IKEA", "Home Depot", "Williams-Sonoma", "Crate and Barrel", "Pottery Barn"],
    "Books": ["Penguin Random House", "HarperCollins", "Scholastic", "Simon & Schuster"],
    "Sports & Outdoors": ["Nike", "Adidas", "Columbia", "The North Face", "Patagonia"],
    "Health & Beauty": ["L'Oréal", "Estée Lauder", "Nivea", "Dove", "Neutrogena"],
    "Toys & Games": ["LEGO", "Mattel", "Hasbro", "Nintendo", "Fisher-Price"],
    "Office Stationery": ["3M", "Staples", "Sharpie", "Post-it", "Bic"]
}

# Generate product data
products = []
for category in categories:
    brand_list = brand_names[category]
    for brand in brand_list:
        for _ in range(100):
            model = fake.word().capitalize() + " " + fake.word().capitalize()  # Generate a two-word model name
            price_range = price_ranges[category]
            price = round(random.uniform(*price_range), 2)
            quantity = random.choice(quantities)
            products.append({
                "Category": category,
                "Product_Name": f"{brand} {model}",
                "Brand": brand,
                "Model": model,
                "Price": price,
                "Quantity": quantity
            })

# Generate unique products by deduplicating based on "Product_Name" and "Category"
unique_products = {product["Product_Name"] + product["Category"]: product for product in products}
products = list(unique_products.values())

# Generate seller data
sellers = []
seller_id = 1
for category in categories:
    num_sellers = random.randint(50, 100)
    for _ in range(num_sellers):
        sellers.append({
            "Seller_ID": seller_id,
            "Seller_Category": category,
            "Seller_Rating": round(random.uniform(3, 5), 2)
        })
        seller_id += 1

uk_cities_mapping = {
    "London": "Greater London",
    "Manchester": "North West",
    "Birmingham": "West Midlands",
    "Liverpool": "North West",
    "Bristol": "South West",
    "Leeds": "Yorkshire and the Humber",
    "Sheffield": "Yorkshire and the Humber",
    "Newcastle": "North East",
    "Southampton": "South East",
    "Oxford": "South East",
    "Glasgow": "Scotland",
    "Edinburgh": "Scotland",
    "Cardiff": "Wales",
    "Belfast": "Northern Ireland",
    "Aberdeen": "Scotland",
    "Plymouth": "South West",
    "Derry": "Northern Ireland",
    # Add more mappings as needed
}

# Step 2: Generate user data and user purchase history
users = []
user_purchase_history = []
user_id = 1
for _ in range(1500):  # Generate extra users to ensure enough unique users
    user_location = random.choice(list(uk_cities_mapping.keys()))
    user_region = uk_cities_mapping[user_location]
    users.append({
        "User_ID": user_id,
        "User_Name": fake.user_name(),
        "Email": fake.email(),
        "Location": user_location,
          "Region": user_region
    })

    num_purchases = random.randint(1, 10)
    purchase_dates = [fake.date_time_this_year() for _ in range(num_purchases)]
    for date in purchase_dates:
        product = random.choice(products)
        seller = random.choice([seller for seller in sellers if seller["Seller_Category"] == product["Category"]])
        rating = min(max(round(np.random.normal(4.5, 0.5), 2), 1), 5)
        review = fake.paragraph() if random.random() > 0.4 else None
        user_purchase_history.append({
            "User_ID": user_id,
            "Product": product["Product_Name"],
            "Category": product["Category"],
            "Seller_ID": seller["Seller_ID"],
            "Purchase_Date": date,
            "Price": product["Price"],
            "Quantity": product["Quantity"],
            "Rating": rating,
            "Review": review
        })
    user_id += 1

# Introduce missing values in ratings and reviews
for purchase in user_purchase_history:
    if random.random() < 0.15:
        purchase["Rating"] = None
    if random.random() < 0.4:
        purchase["Review"] = None

# Introduce price outliers
for product in products:
    if product["Category"] in ["Electronics", "Sports & Outdoors"] and random.random() < 0.05:
        product["Price"] *= random.uniform(1.5, 3)

# Create DataFrames
products_df = pd.DataFrame(products)
sellers_df = pd.DataFrame(sellers)
users_df = pd.DataFrame(users)
purchase_history_df = pd.DataFrame(user_purchase_history)

purchase_history_df.rename(columns={'Product': 'Product_Name'}, inplace=True)

merged_df=[]
# Merge DataFrames
merged_df = purchase_history_df.merge(products_df, on=["Product_Name", "Category"], how="left")
merged_df = merged_df.merge(sellers_df, on="Seller_ID", how="left")
merged_df = merged_df.merge(users_df, on="User_ID", how="left")

merged_df = merged_df.drop(["Price_y", "Quantity_y"], axis=1)

# Rename columns if needed
merged_df.rename(columns={"Price_x": "Price", "Quantity_x": "Quantity"}, inplace=True)

# Save the dataset to a CSV file
merged_df.to_csv("e-commerce_dataset.csv", index=False)

print(purchase_history_df.head())
print(products_df.head())

total_quantity_sold = purchase_history_df.groupby(["Product_Name", "Seller_ID"])["Quantity"].sum().reset_index()

# Normalize total quantities to [0, 1] for each product name
total_quantity_sold["Normalized_Quantity"] = total_quantity_sold.groupby("Product_Name")["Quantity"].transform(lambda x: x / x.sum())

# Assign ratings based on normalized quantities
total_quantity_sold["Product_Rating"] = total_quantity_sold["Normalized_Quantity"].apply(lambda x: round(4 + x, 2))  # Higher rating for higher quantity

# Merge product ratings with products_df
products_df = products_df.merge(total_quantity_sold[["Product_Name", "Product_Rating"]], on=["Product_Name"], how="left")

# Fill missing ratings with a default value
products_df["Product_Rating"].fillna(4.5, inplace=True)

# Display the first few rows of products_df with ratings
print(products_df.head())

# Merge merged_df and products_df again
final_merged_df = merged_df.merge(products_df, on=["Product_Name", "Category"], how="left")

# Drop duplicate columns from products_df
final_merged_df.drop(["Price_y", "Quantity_y", "Product_Rating_y"], axis=1, inplace=True)

# Rename columns if needed
final_merged_df.rename(columns={
    "Price_x": "Price",
    "Quantity_x": "Quantity",
    "Product_Rating_x": "Product_Rating"
}, inplace=True)

# Display the first few rows of the final merged dataset
print(final_merged_df.head())

final_merged_df.info()

# Drop duplicate columns from the merged dataframe
final_merged_df.drop(["Price_y", "Quantity_y", "Brand_y"], axis=1, inplace=True)

# Rename columns if needed
final_merged_df.rename(columns={
    "Price_x": "Price",
    "Quantity_x": "Quantity",
    "Brand_x": "Brand",
    "Model_x": "Model"
}, inplace=True)

# Display the first few rows of the final merged dataset
final_merged_df.head()
final_merged_df.to_csv("../../data/raw/e-commerce_dataset.csv", index=False)
final_merged_df.columns