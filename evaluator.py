import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import time

class ModelEvaluator:
    """
    Evaluation framework for recommendation systems
    """
    
    def __init__(self, recommendation_engine):
        self.rec_engine = recommendation_engine
        self.evaluation_results = {}
    
    def evaluate_content_based_model(self, test_size=0.2, n_recommendations=10):
        """
        Evaluate content-based recommendation model
        """
        # Create test scenarios
        test_data = self._create_test_scenarios(test_size)
        
        results = {
            'precision_at_k': [],
            'recall_at_k': [],
            'response_times': []
        }
        
        for scenario in test_data:
            start_time = time.time()
            
            # Get recommendations
            recommendations = self.rec_engine.get_content_based_recommendations(
                preferred_genres=scenario['preferred_genres'],
                content_types=scenario['content_types'],
                n_recommendations=n_recommendations
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Calculate metrics
            precision = self._calculate_precision_at_k(
                recommendations, scenario['relevant_items'], n_recommendations
            )
            recall = self._calculate_recall_at_k(
                recommendations, scenario['relevant_items'], n_recommendations
            )
            
            results['precision_at_k'].append(precision)
            results['recall_at_k'].append(recall)
            results['response_times'].append(response_time)
        
        # Calculate average metrics
        avg_precision = np.mean(results['precision_at_k'])
        avg_recall = np.mean(results['recall_at_k'])
        avg_response_time = np.mean(results['response_times'])
        
        self.evaluation_results['content_based'] = {
            'precision_at_k': avg_precision,
            'recall_at_k': avg_recall,
            'f1_score': 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0,
            'avg_response_time_ms': avg_response_time
        }
        
        return self.evaluation_results['content_based']
    
    def evaluate_collaborative_model(self, n_test_users=100, n_recommendations=10):
        """
        Evaluate collaborative filtering model
        """
        results = {
            'precision_at_k': [],
            'recall_at_k': [],
            'response_times': []
        }
        
        # Create synthetic test users
        test_users = self._create_synthetic_test_users(n_test_users)
        
        for user_ratings in test_users:
            start_time = time.time()
            
            # Get recommendations
            recommendations = self.rec_engine.get_collaborative_recommendations(
                user_ratings=user_ratings,
                n_recommendations=n_recommendations
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            # Create relevant items based on highly rated content
            relevant_items = [title for title, rating in user_ratings.items() if rating > 7]
            
            # Calculate metrics
            precision = self._calculate_precision_at_k(
                recommendations, relevant_items, n_recommendations
            )
            recall = self._calculate_recall_at_k(
                recommendations, relevant_items, n_recommendations
            )
            
            results['precision_at_k'].append(precision)
            results['recall_at_k'].append(recall)
            results['response_times'].append(response_time)
        
        # Calculate average metrics
        avg_precision = np.mean(results['precision_at_k'])
        avg_recall = np.mean(results['recall_at_k'])
        avg_response_time = np.mean(results['response_times'])
        
        self.evaluation_results['collaborative'] = {
            'precision_at_k': avg_precision,
            'recall_at_k': avg_recall,
            'f1_score': 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0,
            'avg_response_time_ms': avg_response_time
        }
        
        return self.evaluation_results['collaborative']
    
    def evaluate_diversity(self, recommendations_list):
        """
        Evaluate diversity of recommendations
        """
        # Calculate genre diversity
        all_genres = []
        for recommendations in recommendations_list:
            for _, row in recommendations.iterrows():
                genres = str(row['Genres']).split(', ')
                all_genres.extend(genres)
        
        unique_genres = len(set(all_genres))
        total_recommendations = sum(len(recs) for recs in recommendations_list)
        
        diversity_score = unique_genres / total_recommendations if total_recommendations > 0 else 0
        
        return {
            'genre_diversity': diversity_score,
            'unique_genres': unique_genres,
            'total_recommendations': total_recommendations
        }
    
    def evaluate_novelty(self, recommendations, popular_threshold=0.8):
        """
        Evaluate novelty of recommendations (how non-obvious they are)
        """
        # Calculate popularity scores based on IMDB ratings
        df = self.rec_engine.data
        df['imdb_numeric'] = pd.to_numeric(df['Imdb Score'].str.replace('/10', ''), errors='coerce')
        
        # Define popular items as those with high IMDB scores
        popular_items = df[df['imdb_numeric'] >= popular_threshold * 10]['Title'].tolist()
        
        novelty_scores = []
        
        for _, row in recommendations.iterrows():
            if row['Title'] in popular_items:
                novelty_scores.append(0)  # Not novel (popular item)
            else:
                novelty_scores.append(1)  # Novel (less popular item)
        
        avg_novelty = np.mean(novelty_scores) if novelty_scores else 0
        
        return {
            'novelty_score': avg_novelty,
            'novel_items': sum(novelty_scores),
            'total_items': len(recommendations)
        }
    
    def evaluate_coverage(self, all_recommendations):
        """
        Evaluate catalog coverage (what percentage of items can be recommended)
        """
        all_recommended_items = set()
        
        for recommendations in all_recommendations:
            recommended_titles = recommendations['Title'].tolist()
            all_recommended_items.update(recommended_titles)
        
        total_items = len(self.rec_engine.data)
        coverage = len(all_recommended_items) / total_items
        
        return {
            'coverage_score': coverage,
            'recommended_items': len(all_recommended_items),
            'total_items': total_items
        }
    
    def _calculate_precision_at_k(self, recommendations, relevant_items, k):
        """Calculate precision@k metric"""
        if len(recommendations) == 0:
            return 0
        
        recommended_titles = recommendations['Title'].head(k).tolist()
        relevant_recommended = len(set(recommended_titles) & set(relevant_items))
        
        return relevant_recommended / min(k, len(recommended_titles))
    
    def _calculate_recall_at_k(self, recommendations, relevant_items, k):
        """Calculate recall@k metric"""
        if len(relevant_items) == 0:
            return 0
        
        recommended_titles = recommendations['Title'].head(k).tolist()
        relevant_recommended = len(set(recommended_titles) & set(relevant_items))
        
        return relevant_recommended / len(relevant_items)
    
    def _create_test_scenarios(self, test_size):
        """Create test scenarios for content-based evaluation"""
        df = self.rec_engine.data
        
        # Get all unique genres
        all_genres = df['Genres'].str.split(', ').explode().unique()
        all_genres = [g for g in all_genres if pd.notna(g)]
        
        # Create test scenarios
        scenarios = []
        n_scenarios = int(len(df) * test_size)
        
        np.random.seed(42)
        
        for _ in range(min(n_scenarios, 50)):  # Limit to 50 scenarios for efficiency
            # Random genre preferences
            n_preferred_genres = np.random.randint(1, 4)
            preferred_genres = np.random.choice(all_genres, n_preferred_genres, replace=False).tolist()
            
            # Random content types
            content_types = np.random.choice(['Movie', 'TV Show'], np.random.randint(1, 3), replace=False).tolist()
            
            # Find relevant items (high-rated items in preferred genres)
            relevant_items = df[
                (df['Genres'].str.contains('|'.join(preferred_genres), na=False)) &
                (df['Content Type'].isin(content_types)) &
                (pd.to_numeric(df['Imdb Score'].str.replace('/10', ''), errors='coerce') >= 7.0)
            ]['Title'].tolist()
            
            scenarios.append({
                'preferred_genres': preferred_genres,
                'content_types': content_types,
                'relevant_items': relevant_items
            })
        
        return scenarios
    
    def _create_synthetic_test_users(self, n_users):
        """Create synthetic test users for collaborative filtering evaluation"""
        df = self.rec_engine.data
        all_titles = df['Title'].tolist()
        
        test_users = []
        np.random.seed(42)
        
        for _ in range(min(n_users, 20)):  # Limit for efficiency
            # Each user rates 5-15 items
            n_ratings = np.random.randint(5, 16)
            rated_titles = np.random.choice(all_titles, n_ratings, replace=False)
            
            # Generate ratings (biased towards higher ratings)
            ratings = np.random.choice(range(1, 11), n_ratings, p=[0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05])
            
            user_ratings = dict(zip(rated_titles, ratings))
            test_users.append(user_ratings)
        
        return test_users
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        report = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models_evaluated': list(self.evaluation_results.keys()),
            'summary': {}
        }
        
        # Add detailed results
        report['detailed_results'] = self.evaluation_results
        
        # Calculate summary statistics
        if 'content_based' in self.evaluation_results:
            cb_results = self.evaluation_results['content_based']
            report['summary']['content_based'] = {
                'overall_score': (cb_results['precision_at_k'] + cb_results['recall_at_k']) / 2,
                'performance_grade': self._get_performance_grade((cb_results['precision_at_k'] + cb_results['recall_at_k']) / 2)
            }
        
        if 'collaborative' in self.evaluation_results:
            cf_results = self.evaluation_results['collaborative']
            report['summary']['collaborative'] = {
                'overall_score': (cf_results['precision_at_k'] + cf_results['recall_at_k']) / 2,
                'performance_grade': self._get_performance_grade((cf_results['precision_at_k'] + cf_results['recall_at_k']) / 2)
            }
        
        return report
    
    def _get_performance_grade(self, score):
        """Convert numeric score to performance grade"""
        if score >= 0.8:
            return 'Excellent'
        elif score >= 0.6:
            return 'Good'
        elif score >= 0.4:
            return 'Fair'
        else:
            return 'Needs Improvement'
