import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

def train_xgb_model():
    print("Loading anime.csv...")
    anime = pd.read_csv("anime.csv")
    anime = anime.dropna(subset=['name'])
    
    # Fill NAs
    anime['genre'] = anime['genre'].fillna('')
    anime['type'] = anime['type'].fillna('Unknown')
    anime['rating'] = anime['rating'].fillna(anime['rating'].mean())
    anime['members'] = anime['members'].fillna(0)
    
    print("Preparing features for XGBoost...")
    # TF-IDF on Genres
    tfidf = TfidfVectorizer(max_features=50) # Top 50 genres
    genre_features = tfidf.fit_transform(anime['genre']).toarray()
    
    # Label Encode Type
    le = LabelEncoder()
    type_encoded = le.fit_transform(anime['type'])
    
    # Create Feature Matrix X and Target Y
    # We will train XGBoost to predict the 'rating' based on genres, type, and members.
    # The true magic is that we will extract the XGBoost tree leaf indices to use as an Advanced Embedding!
    X = np.column_stack((genre_features, type_encoded, anime['members'].values))
    y = anime['rating'].values
    
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    print("Extracting XGBoost Leaf Embeddings (Advanced Feature Engineering)...")
    # Instead of just predicting, we apply the model to get the leaf index for each anime across all 100 trees.
    # This creates a dense, non-linear embedding space of shape (num_anime, 100)
    xgb_embeddings = model.apply(X)
    
    print("Computing Cosine Similarity on XGBoost Embeddings...")
    # Calculate similarity
    similarity_matrix = cosine_similarity(xgb_embeddings)
    
    print("Extracting top 50 matches for each anime...")
    xgb_model_dict = {}
    
    for idx, row in anime.iterrows():
        anime_id = int(row['anime_id'])
        sim_scores = similarity_matrix[idx]
        
        # Get top 51 (including itself)
        top_indices = np.argpartition(sim_scores, -51)[-51:]
        top_indices = top_indices[np.argsort(sim_scores[top_indices])[::-1]]
        
        similar_animes = {}
        for top_idx in top_indices:
            if top_idx != idx:
                similar_anime_id = int(anime.iloc[top_idx]['anime_id'])
                score = round(float(sim_scores[top_idx]), 4)
                similar_animes[str(similar_anime_id)] = score
                
        xgb_model_dict[str(anime_id)] = similar_animes

    print("Saving xgb_model.json...")
    with open("xgb_model.json", "w") as f:
        json.dump(xgb_model_dict, f)
        
    print("Done! Offline XGBoost training complete. Model saved to xgb_model.json")

if __name__ == "__main__":
    train_xgb_model()
