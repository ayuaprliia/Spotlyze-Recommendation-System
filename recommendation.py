from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
import json

app = FastAPI()

# Load the dataset
df = pd.read_csv('Dataset/recommendation_dataset/skincare_dataset_fix.csv')

# Filter dataset to include only relevant product categories 
df2 = df[df['category'].isin(['Cleanser', 'Moisturizer', 'Mask', 'Treatment'])]
df2['skin_type'] = df2['skin_type'].apply(lambda x: re.sub(r'\.', '', str(x)))

# Define features used for one-hot encoding
features = ['all skin types', 'normal skin', 'dry skin', 'oily', 'combination', 'acne', 'sensitive', 'wrinkles',
            'dark circle', 'skin brightness', 'uneven skin texture', 'skin dullness', 'hydration and nourishment', 'general care']

one_hot_encodings = np.zeros((len(df2), len(features)))

# Cosine similarity-based recommendation function
def recs_cs(vector=None, product_name=None, category=None, count=8):
    products = []
    if product_name:
        idx = name2index(product_name)
        fv = one_hot_encodings[idx]
    elif vector is not None:
        fv = vector
    else:
        return []

    # Calculate cosine similarity
    cs_values = cosine_similarity(np.array([fv]), one_hot_encodings)
    df2['cs'] = cs_values[0]

     # Apply category filtering 
    if category:
        dff = df2[df2['category'] == category]
    else:
        dff = df2

    # Exclude the input product from recommendations
    if product_name:
        dff = dff[dff['product_name'] != product_name]

    # Sort by cosine similarity and get top recommendations
    recommendations = dff.sort_values('cs', ascending=False).head(count)
    data = recommendations[['product_name', 'product_brand', 'ingredients', 'skin_type', 'concern', 'price', 'product_image_url']].to_dict('split')['data']
    
    for element in data:
        products.append(wrap(element))
    return products

# Helper function to format the recommended product data
def wrap(info_arr):
    return {
        'product_name': info_arr[0],
        'product_brand': info_arr[1],
        'price': info_arr[5],
        'skin_type': info_arr[3],
        'product_image_url': info_arr[6],
        'ingredients': info_arr[2]
    }

# Pydantic model for the user's input
class SkinCareRequest(BaseModel):
    skin_type: str
    skin_sensitivity: str
    concerns: list

@app.post("/recommend_products")
async def recommend_products(request: SkinCareRequest):
    # Convert user input into the feature vector
    vector = np.zeros(len(features))

    # Set skin type
    if request.skin_type == 'all':
        vector[:5] = 1
    else:
        if request.skin_type in features[:5]:
            vector[features.index(request.skin_type)] = 1

    # Set skin sensitivity
    if request.skin_sensitivity == 'sensitive':
        vector[features.index('sensitive')] = 1

    # Set concerns
    for concern in request.concerns:
        if concern in features:
            vector[features.index(concern)] = 1

    # Get recommendations for each category
    categories = df2['category'].unique()
    category_recommendations = {}

    for category in categories:
        category_recommendations[category] = recs_cs(vector=vector, category=category, count=8)
    
    return {"recommendations": category_recommendations}