import pandas as pd
import numpy as np
import re
from datetime import datetime

class DataProcessor:
    """
    Data preprocessing and analysis for Netflix dataset
    """
    
    def __init__(self):
        pass
    
    def preprocess_data(self, df):
        """
        Comprehensive data preprocessing
        """
        # Make a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Clean column names
        processed_df.columns = processed_df.columns.str.strip()
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Clean text fields
        processed_df = self._clean_text_fields(processed_df)
        
        # Process dates
        processed_df = self._process_dates(processed_df)
        
        # Extract numeric ratings
        processed_df = self._process_ratings(processed_df)
        
        # Clean duration field
        processed_df = self._process_duration(processed_df)
        
        # Process genres
        processed_df = self._process_genres(processed_df)
        
        return processed_df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        
        # Fill missing values with appropriate defaults
        df['Director'] = df['Director'].fillna('Unknown Director')
        df['Cast'] = df['Cast'].fillna('Unknown Cast')
        df['Genres'] = df['Genres'].fillna('Unknown Genre')
        df['Production Country'] = df['Production Country'].fillna('Unknown Country')
        df['Description'] = df['Description'].fillna('No description available')
        df['Date Added'] = df['Date Added'].fillna('Unknown')
        
        return df
    
    def _clean_text_fields(self, df):
        """Clean text fields"""
        
        text_columns = ['Title', 'Description', 'Director', 'Cast', 'Genres']
        
        for col in text_columns:
            if col in df.columns:
                # Remove extra whitespaces
                df[col] = df[col].astype(str).str.strip()
                
                # Remove duplicate commas and clean separators
                df[col] = df[col].str.replace(r',\s*,', ',', regex=True)
                df[col] = df[col].str.replace(r'^\s*,\s*|\s*,\s*$', '', regex=True)
        
        return df
    
    def _process_dates(self, df):
        """Process date fields"""
        
        # Convert Release Date to numeric year
        if 'Release Date' in df.columns:
            df['Release Date'] = pd.to_numeric(df['Release Date'], errors='coerce')
            
            # Filter out unrealistic years
            df = df[(df['Release Date'] >= 1900) & (df['Release Date'] <= 2025)]
        
        # Process Date Added
        if 'Date Added' in df.columns:
            # Try to parse date added
            df['Date Added Parsed'] = pd.to_datetime(df['Date Added'], errors='coerce')
            df['Year Added'] = df['Date Added Parsed'].dt.year
        
        return df
    
    def _process_ratings(self, df):
        """Process rating fields"""
        
        # Clean IMDB Score
        if 'Imdb Score' in df.columns:
            # Extract numeric part from IMDB score
            df['Imdb Score Numeric'] = df['Imdb Score'].str.extract(r'(\d+\.?\d*)').astype(float)
            
            # Filter out unrealistic scores
            df = df[(df['Imdb Score Numeric'] >= 0) & (df['Imdb Score Numeric'] <= 10)]
        
        return df
    
    def _process_duration(self, df):
        """Process duration field"""
        
        if 'Duration' in df.columns:
            # Extract minutes for movies
            df['Duration Minutes'] = df['Duration'].str.extract(r'(\d+) min').astype(float)
            
            # Extract seasons for TV shows
            df['Duration Seasons'] = df['Duration'].str.extract(r'(\d+) Season').astype(float)
            
            # Create unified duration metric (convert seasons to approximate minutes)
            df['Duration Unified'] = df['Duration Minutes'].fillna(
                df['Duration Seasons'] * 600  # Assume 10 hours per season
            )
        
        return df
    
    def _process_genres(self, df):
        """Process genres field"""
        
        if 'Genres' in df.columns:
            # Split genres and count
            df['Genre Count'] = df['Genres'].str.count(',') + 1
            
            # Create genre flags for popular genres
            popular_genres = [
                'Drama', 'Comedy', 'Action', 'Thriller', 'Horror',
                'Romance', 'Documentary', 'Crime', 'Adventure', 'Sci-Fi'
            ]
            
            for genre in popular_genres:
                df[f'Is_{genre}'] = df['Genres'].str.contains(genre, case=False, na=False)
        
        return df
    
    def get_data_summary(self, df):
        """Get comprehensive data summary"""
        
        summary = {
            'total_records': len(df),
            'total_movies': len(df[df['Content Type'] == 'Movie']),
            'total_tv_shows': len(df[df['Content Type'] == 'TV Show']),
            'date_range': {
                'earliest': df['Release Date'].min(),
                'latest': df['Release Date'].max()
            },
            'top_genres': df['Genres'].str.split(', ').explode().value_counts().head(10).to_dict(),
            'top_countries': df['Production Country'].str.split(', ').explode().value_counts().head(10).to_dict(),
            'rating_distribution': df['Rating'].value_counts().to_dict(),
            'average_imdb_score': df['Imdb Score Numeric'].mean()
        }
        
        return summary
    
    def detect_data_quality_issues(self, df):
        """Detect potential data quality issues"""
        
        issues = []
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['Title', 'Release Date']).sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} potential duplicate titles")
        
        # Check for missing critical fields
        critical_fields = ['Title', 'Content Type', 'Release Date']
        for field in critical_fields:
            if field in df.columns:
                missing = df[field].isna().sum()
                if missing > 0:
                    issues.append(f"Missing {missing} values in {field}")
        
        # Check for unrealistic IMDB scores
        if 'Imdb Score Numeric' in df.columns:
            invalid_scores = ((df['Imdb Score Numeric'] < 0) | (df['Imdb Score Numeric'] > 10)).sum()
            if invalid_scores > 0:
                issues.append(f"Found {invalid_scores} invalid IMDB scores")
        
        # Check for unrealistic release dates
        if 'Release Date' in df.columns:
            current_year = datetime.now().year
            invalid_years = ((df['Release Date'] < 1900) | (df['Release Date'] > current_year + 5)).sum()
            if invalid_years > 0:
                issues.append(f"Found {invalid_years} invalid release years")
        
        return issues
    
    def create_feature_matrix(self, df):
        """Create feature matrix for machine learning"""
        
        # Select numeric features
        numeric_features = ['Release Date', 'Imdb Score Numeric', 'Duration Unified', 'Genre Count']
        
        # Select categorical features
        categorical_features = ['Content Type', 'Rating']
        
        # Create feature matrix
        feature_df = df[numeric_features + categorical_features].copy()
        
        # One-hot encode categorical variables
        feature_df = pd.get_dummies(feature_df, columns=categorical_features)
        
        # Fill missing values with median/mode
        for col in feature_df.columns:
            if feature_df[col].dtype in ['int64', 'float64']:
                feature_df[col].fillna(feature_df[col].median(), inplace=True)
            else:
                feature_df[col].fillna(feature_df[col].mode()[0], inplace=True)
        
        return feature_df
    
    def extract_content_keywords(self, df, n_keywords=100):
        """Extract most common keywords from descriptions"""
        
        from collections import Counter
        import re
        
        # Combine all descriptions
        all_descriptions = ' '.join(df['Description'].astype(str))
        
        # Clean and extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_descriptions.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'through', 'during',
            'before', 'after', 'above', 'below', 'between'
        }
        
        words = [word for word in words if word not in stop_words]
        
        # Count and return top keywords
        word_counts = Counter(words)
        
        return word_counts.most_common(n_keywords)
