import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import re

class RecommendationEngine:
    """
    Netflix content recommendation engine with multiple algorithms
    """
    
    def __init__(self, data):
        self.data = data
        self.tfidf_vectorizer = None
        self.content_similarity_matrix = None
        self.svd_model = None
        self._prepare_models()
    
    def _prepare_models(self):
        """Prepare recommendation models"""
        self._prepare_content_based_model()
        self._prepare_collaborative_model()
    
    def _prepare_content_based_model(self):
        """Prepare content-based filtering model"""
        # Create content features
        self.data['combined_features'] = (
            self.data['Genres'].fillna('') + ' ' +
            self.data['Cast'].fillna('') + ' ' +
            self.data['Director'].fillna('') + ' ' +
            self.data['Description'].fillna('')
        )
        
        # Clean and preprocess text
        self.data['combined_features'] = self.data['combined_features'].apply(self._clean_text)
        
        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.data['combined_features'])
        
        # Calculate similarity matrix
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
    
    def _prepare_collaborative_model(self):
        """Prepare collaborative filtering model using SVD"""
        # Create user-item matrix simulation
        # Since we don't have actual user data, we'll create synthetic interactions
        self._create_synthetic_user_data()
        
        # Apply SVD for dimensionality reduction
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        self.user_item_matrix_svd = self.svd_model.fit_transform(self.synthetic_user_item_matrix)
    
    def _create_synthetic_user_data(self):
        """Create synthetic user-item interaction matrix"""
        n_users = 1000
        n_items = len(self.data)
        
        # Create sparse user-item matrix
        np.random.seed(42)
        self.synthetic_user_item_matrix = np.random.exponential(scale=2, size=(n_users, n_items))
        
        # Make it sparse (most users haven't watched most content)
        mask = np.random.random((n_users, n_items)) < 0.1
        self.synthetic_user_item_matrix = self.synthetic_user_item_matrix * mask
    
    def _clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def get_content_based_recommendations(self, preferred_genres=None, content_types=None, n_recommendations=10):
        """
        Get content-based recommendations
        """
        filtered_data = self.data.copy()
        
        # Filter by content type
        if content_types:
            filtered_data = filtered_data[filtered_data['Content Type'].isin(content_types)]
        
        # Filter by genres
        if preferred_genres:
            genre_filter = '|'.join(preferred_genres)
            filtered_data = filtered_data[
                filtered_data['Genres'].str.contains(genre_filter, case=False, na=False)
            ]
        
        # Sort by IMDB score and return top recommendations
        filtered_data['imdb_numeric'] = pd.to_numeric(
            filtered_data['Imdb Score'].str.replace('/10', ''), 
            errors='coerce'
        )
        
        recommendations = filtered_data.nlargest(n_recommendations, 'imdb_numeric')
        
        return recommendations[['Title', 'Content Type', 'Genres', 'Rating', 'Imdb Score', 'Description']]
    
    def get_collaborative_recommendations(self, user_ratings, n_recommendations=10):
        """
        Get collaborative filtering recommendations
        """
        # Create user profile based on ratings
        user_profile = np.zeros(len(self.data))
        
        for title, rating in user_ratings.items():
            if rating > 5:  # Only consider positively rated content
                indices = self.data[self.data['Title'] == title].index
                if len(indices) > 0:
                    user_profile[indices[0]] = rating / 10.0
        
        # Find similar content using content similarity
        similar_scores = np.dot(self.content_similarity_matrix, user_profile)
        
        # Get top recommendations
        recommended_indices = np.argsort(similar_scores)[::-1]
        
        # Filter out already rated content
        rated_titles = set(user_ratings.keys())
        recommendations = []
        
        for idx in recommended_indices:
            if len(recommendations) >= n_recommendations:
                break
            
            title = self.data.iloc[idx]['Title']
            if title not in rated_titles:
                recommendations.append(idx)
        
        return self.data.iloc[recommendations][['Title', 'Content Type', 'Genres', 'Rating', 'Imdb Score', 'Description']]
    
    def get_similar_content(self, title, n_recommendations=10):
        """
        Get content similar to a specific title
        """
        try:
            # Find the index of the given title
            idx = self.data[self.data['Title'] == title].index[0]
            
            # Get similarity scores for this content
            similarity_scores = list(enumerate(self.content_similarity_matrix[idx]))
            
            # Sort by similarity score
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar content (excluding the input title itself)
            similar_indices = [i[0] for i in similarity_scores[1:n_recommendations+1]]
            
            return self.data.iloc[similar_indices][['Title', 'Content Type', 'Genres', 'Rating', 'Imdb Score', 'Description']]
        
        except IndexError:
            return pd.DataFrame()
    
    def get_trending_content(self, n_recommendations=10):
        """
        Get trending/popular content based on IMDB scores
        """
        # Convert IMDB scores to numeric
        self.data['imdb_numeric'] = pd.to_numeric(
            self.data['Imdb Score'].str.replace('/10', ''), 
            errors='coerce'
        )
        
        # Get highest rated content
        trending = self.data.nlargest(n_recommendations, 'imdb_numeric')
        
        return trending[['Title', 'Content Type', 'Genres', 'Rating', 'Imdb Score', 'Description']]
    
    def get_recommendations_by_genre(self, genre, n_recommendations=10):
        """
        Get top recommendations for a specific genre
        """
        genre_content = self.data[
            self.data['Genres'].str.contains(genre, case=False, na=False)
        ]
        
        # Convert IMDB scores to numeric
        genre_content['imdb_numeric'] = pd.to_numeric(
            genre_content['Imdb Score'].str.replace('/10', ''), 
            errors='coerce'
        )
        
        # Get top rated content in this genre
        recommendations = genre_content.nlargest(n_recommendations, 'imdb_numeric')
        
        return recommendations[['Title', 'Content Type', 'Genres', 'Rating', 'Imdb Score', 'Description']]
