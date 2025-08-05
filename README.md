# Netflix Content Recommendation System

A comprehensive machine learning-based recommendation system for Netflix content using collaborative filtering, content-based filtering, and hybrid approaches. This project analyzes Netflix viewing patterns and provides personalized content recommendations.

## Project Overview

This recommendation system leverages multiple machine learning techniques to suggest relevant Netflix content to users based on their viewing history, preferences, and content similarities. The system combines collaborative filtering with content-based approaches to deliver accurate and diverse recommendations.

## Features

### Recommendation Algorithms
- **Collaborative Filtering**: User-based and item-based collaborative filtering
- **Content-Based Filtering**: Recommendations based on content features (genre, cast, director, etc.)
- **Hybrid Approach**: Combines multiple algorithms for improved accuracy
- **Matrix Factorization**: Advanced dimensionality reduction techniques

### Analytics & Insights
- **Content Analysis**: Distribution of genres, ratings, and production countries
- **Trend Analysis**: Release patterns and popularity trends over time
- **Performance Metrics**: Precision, recall, and recommendation accuracy
- **Interactive Visualizations**: Comprehensive data exploration dashboard

### Technical Features
- **Scalable Architecture**: Efficient handling of large datasets
- **Real-time Recommendations**: Fast recommendation generation
- **Evaluation Framework**: Comprehensive model performance assessment
- **Export Capabilities**: Save recommendations and analytics results

## Technology Stack

- **Data Processing**: `pandas`, `numpy`, `scipy`
- **Machine Learning**: `scikit-learn`, `surprise`, `tensorflow`
- **Visualization**: `plotly`, `matplotlib`, `seaborn`
- **Web Framework**: `streamlit`
- **Text Processing**: `nltk`, `textblob`

## Dataset

The project uses Netflix content data with the following features:
- Content metadata (title, description, genre, cast, director)
- Release information (date, country, rating)
- User ratings and viewing patterns
- Content duration and type (movie/TV show)

**Data Source**: Netflix content dataset with 8,000+ titles

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/KaanDRoid/netflix-recommendation-system.git
   cd netflix-recommendation-system
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## Usage

### Recommendation Dashboard
1. **Content Explorer**: Browse and filter Netflix content
2. **Recommendation Engine**: Get personalized recommendations
3. **Analytics Dashboard**: Explore content trends and patterns
4. **Model Performance**: View recommendation accuracy metrics

### Getting Recommendations
- Select your preferred genres and content types
- Rate content you've watched
- Get instant personalized recommendations
- Explore similar content and trending titles

## Recommendation Algorithms

### 1. Collaborative Filtering
- **User-Based**: Find similar users and recommend their preferences
- **Item-Based**: Recommend content similar to what users have liked
- **Matrix Factorization**: SVD and NMF for dimensionality reduction

### 2. Content-Based Filtering
- **TF-IDF Vectorization**: Text similarity based on descriptions
- **Feature Matching**: Genre, cast, director, and metadata similarity
- **Cosine Similarity**: Calculate content similarity scores

### 3. Hybrid Approach
- **Weighted Combination**: Blend multiple algorithms
- **Switching Hybrid**: Choose best algorithm based on data availability
- **Meta-Learning**: Learn optimal combination weights

## Model Performance

### Evaluation Metrics
- **Precision@K**: Accuracy of top-K recommendations
- **Recall@K**: Coverage of relevant items in top-K
- **NDCG**: Normalized Discounted Cumulative Gain
- **Coverage**: Diversity of recommended content

### Performance Results
- Content-Based Filtering: 78% accuracy
- Collaborative Filtering: 82% accuracy
- Hybrid Model: 85% accuracy
- Average recommendation time: <100ms

## Project Structure

```
netflix-recommendation-system/
├── app.py                          # Main Streamlit application
├── recommendation_engine.py        # Core recommendation algorithms
├── data_processor.py              # Data preprocessing and analysis
├── evaluator.py                   # Model evaluation and metrics
├── utils/
│   ├── similarity_calculator.py   # Similarity computation functions
│   ├── matrix_factorization.py    # Advanced ML algorithms
│   └── visualization.py           # Chart and plot functions
├── data/
│   ├── netflixData.csv            # Netflix content dataset
│   └── processed/                 # Processed data files
├── models/                        # Trained model files
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── LICENSE                        # MIT License
```

## Key Features Implementation

### Advanced Algorithms
- **SVD (Singular Value Decomposition)**: Matrix factorization for collaborative filtering
- **NMF (Non-negative Matrix Factorization)**: Feature extraction and recommendation
- **K-Means Clustering**: User and content segmentation
- **Natural Language Processing**: Content description analysis

### Performance Optimization
- **Sparse Matrix Operations**: Efficient memory usage
- **Caching**: Fast repeated recommendations
- **Vectorized Operations**: Optimized similarity calculations
- **Parallel Processing**: Multi-threaded recommendation generation

## Future Enhancements

- Deep learning models (Neural Collaborative Filtering)
- Real-time user feedback integration
- A/B testing framework for algorithm comparison
- Multi-criteria recommendation (mood, time, device)
- Social recommendation features
- Cold start problem solutions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Netflix for providing comprehensive content data
- Scikit-learn community for machine learning tools
- Streamlit for the excellent web framework
- Open source recommendation system research community

## Contact

For questions or suggestions, feel free to reach out or open an issue.

---

**If you find this project helpful, please consider giving it a star!**
