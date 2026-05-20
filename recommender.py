import pandas as pd
import numpy as np
import re
import string
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class AnimeRecommender:
    def __init__(self, data_path="anime.csv", cf_path="cf_model.json", xgb_path="xgb_model.json"):
        self.data_path = data_path
        self.cf_path = cf_path
        self.xgb_path = xgb_path
        self.anime_data = None
        self.tfidf_matrix = None
        self.tfidf = None
        self.indices = None
        self.cf_model = {}
        self.xgb_model = {}
        self._load_and_preprocess()

    def text_cleaning(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'&quot;', '', text)
        text = "".join([char for char in text if char not in string.punctuation])
        text = re.sub(r'.hack//', '', text)
        text = re.sub(r'&#039;', '', text)
        text = re.sub(r'A&#039;s', '', text)
        text = re.sub(r'I&#039;', 'I\'', text)
        text = re.sub(r'&amp;', 'and', text)
        text = re.sub(r'Â°', '', text)
        return text.lower()

    def _load_and_preprocess(self):
        print("Loading anime data...")
        self.anime_data = pd.read_csv(self.data_path)
        self.anime_data = self.anime_data.dropna(subset=['name'])
        self.anime_data['clean_name'] = self.anime_data['name'].apply(self.text_cleaning)
        
        self.anime_data['genre'] = self.anime_data['genre'].fillna('')
        self.anime_data['type'] = self.anime_data['type'].fillna('Unknown')
        self.anime_data['tags'] = self.anime_data['genre'].str.replace(',', ' ') + " " + self.anime_data['type']
        
        print("Vectorizing TF-IDF (Content-Based) features...")
        self.tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 4), stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.anime_data['tags'])
        self.indices = pd.Series(self.anime_data.index, index=self.anime_data['clean_name']).drop_duplicates()
        
        # Load CF Model
        if os.path.exists(self.cf_path):
            print(f"Loading CF model from {self.cf_path}...")
            with open(self.cf_path, "r") as f:
                self.cf_model = json.load(f)
                
        # Load XGB Model
        if os.path.exists(self.xgb_path):
            print(f"Loading XGBoost model from {self.xgb_path}...")
            with open(self.xgb_path, "r") as f:
                self.xgb_model = json.load(f)
                
        print("Recommender engine initialized!")

    def search_anime(self, query, limit=10):
        if not query:
            return []
        query_clean = self.text_cleaning(query)
        matches = self.anime_data[self.anime_data['clean_name'].str.contains(query_clean, case=False, na=False)]
        matches = matches.sort_values(by='members', ascending=False).head(limit)
        matches = matches.replace({np.nan: None})
        return matches[['anime_id', 'name', 'type', 'rating']].to_dict(orient='records')

    def get_recommendations(self, title, model_type='hybrid', limit=10, sort_by_rating=False):
        title_clean = self.text_cleaning(title)
        
        if title_clean not in self.indices:
            return {"error": "Anime not found"}
            
        idx = self.indices[title_clean]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
            
        target_anime_id = str(self.anime_data.iloc[idx]['anime_id'])
            
        # Base TF-IDF scores
        cosine_sim = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        
        scores = []
        
        # Route by Model Type
        if model_type == 'collaborative':
            cf_similarities = self.cf_model.get(target_anime_id, {})
            # Only include animes that have a CF similarity score
            for i in range(len(self.anime_data)):
                if i == idx: continue
                aid_str = str(self.anime_data.iloc[i]['anime_id'])
                if aid_str in cf_similarities:
                    scores.append((i, cf_similarities[aid_str]))
            
        elif model_type == 'hybrid':
            xgb_similarities = self.xgb_model.get(target_anime_id, {})
            # Only include animes in the precomputed XGB list
            for i in range(len(self.anime_data)):
                if i == idx: continue
                aid_str = str(self.anime_data.iloc[i]['anime_id'])
                if aid_str in xgb_similarities:
                    scores.append((i, xgb_similarities[aid_str]))
                    
        else:
            # 'content' fallback
            for i, cb_score in enumerate(cosine_sim):
                if i == idx: continue
                scores.append((i, cb_score))
        
        # If no results (e.g. model file missing or anime not in matrix), fallback to content
        if not scores:
            for i, cb_score in enumerate(cosine_sim):
                if i == idx: continue
                scores.append((i, cb_score))
        
        # Sort and limit
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:limit]
        
        anime_indices = [i[0] for i in scores]
        similarity_scores = [i[1] for i in scores]
        
        results = self.anime_data.iloc[anime_indices].copy()
        results['similarity'] = similarity_scores
        
        if sort_by_rating:
            results = results.sort_values('rating', ascending=False)
            
        results = results.replace({np.nan: None})
        return results[['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members', 'similarity']].to_dict(orient='records')

    def get_stats(self):
        return {
            "total_anime": len(self.anime_data),
            "avg_rating": round(self.anime_data['rating'].mean(), 2) if not self.anime_data['rating'].empty else 0,
            "top_genres": self.anime_data['genre'].str.split(', ').explode().value_counts().head(5).to_dict()
        }
