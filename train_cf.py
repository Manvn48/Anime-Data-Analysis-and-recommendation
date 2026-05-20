import pandas as pd
import numpy as np
import json
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def train_cf_model():
    print("Loading rating.csv...")
    try:
        ratings = pd.read_csv("rating.csv")
    except FileNotFoundError:
        print("Error: rating.csv not found. Did you extract it?")
        return

    print("Cleaning rating data...")
    # Clean data exactly as done in the original notebook
    ratings['rating'] = ratings['rating'].replace(-1, np.nan)
    ratings = ratings.dropna(subset=['rating'])

    # Map user_id and anime_id to contiguous indices
    user_u = list(sorted(ratings.user_id.unique()))
    item_u = list(sorted(ratings.anime_id.unique()))

    print(f"Total Users: {len(user_u)}, Total Anime: {len(item_u)}")
    
    # Create dictionaries for mapping
    user_to_idx = {user: idx for idx, user in enumerate(user_u)}
    item_to_idx = {item: idx for idx, item in enumerate(item_u)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}

    print("Building sparse interaction matrix...")
    user_idx = ratings['user_id'].map(user_to_idx).values
    item_idx = ratings['anime_id'].map(item_to_idx).values
    rating_values = ratings['rating'].values

    # Sparse matrix: Users as rows, Anime as columns
    sparse_matrix = csr_matrix((rating_values, (user_idx, item_idx)), shape=(len(user_u), len(item_u)))

    print("Running TruncatedSVD (Matrix Factorization)...")
    # 50 latent features is a good standard for SVD on this dataset size
    svd = TruncatedSVD(n_components=50, random_state=42)
    
    # We want anime embeddings, so we fit on the transposed matrix (Anime as rows, Users as columns)
    anime_embeddings = svd.fit_transform(sparse_matrix.T)

    print("Computing cosine similarities and extracting top 50 for each anime...")
    # Calculate similarity between all anime embeddings
    # This results in an (num_anime x num_anime) matrix.
    similarity_matrix = cosine_similarity(anime_embeddings)

    cf_model = {}
    
    # Loop over all anime
    for idx in range(len(item_u)):
        anime_id = int(idx_to_item[idx])
        
        # Get similarities for this anime
        sim_scores = similarity_matrix[idx]
        
        # Get top 50 indices (ignoring the anime itself which has similarity 1.0)
        # Using argpartition for speed
        top_indices = np.argpartition(sim_scores, -51)[-51:]
        top_indices = top_indices[np.argsort(sim_scores[top_indices])[::-1]]
        
        # Build dictionary of {similar_anime_id: score}
        similar_animes = {}
        for top_idx in top_indices:
            if top_idx != idx:
                similar_anime_id = int(idx_to_item[top_idx])
                score = round(float(sim_scores[top_idx]), 4)
                similar_animes[str(similar_anime_id)] = score
                
        # Limit to strictly top 50
        cf_model[str(anime_id)] = similar_animes

    print("Saving cf_model.json...")
    with open("cf_model.json", "w") as f:
        json.dump(cf_model, f)
        
    print("Done! Offline training complete. Model saved to cf_model.json")

if __name__ == "__main__":
    train_cf_model()
