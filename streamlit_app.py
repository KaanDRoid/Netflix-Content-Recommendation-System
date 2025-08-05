import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import re
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #E50914;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #E50914;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #E50914;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess Netflix data"""
    try:
        df = pd.read_csv('netflixData.csv')
        
        # Basic preprocessing
        df = df.dropna(subset=['Title', 'Content Type'])
        df['Genres'] = df['Genres'].fillna('Unknown')
        df['Cast'] = df['Cast'].fillna('Unknown')
        df['Director'] = df['Director'].fillna('Unknown')
        df['Description'] = df['Description'].fillna('No description available')
        df['Production Country'] = df['Production Country'].fillna('Unknown')
        df['Rating'] = df['Rating'].fillna('Unrated')
        
        # Clean IMDB scores
        df['Imdb_Numeric'] = pd.to_numeric(df['Imdb Score'].str.replace('/10', ''), errors='coerce')
        df['Imdb_Numeric'] = df['Imdb_Numeric'].fillna(df['Imdb_Numeric'].median())
        
        # Clean release dates
        df['Release Date'] = pd.to_numeric(df['Release Date'], errors='coerce')
        df = df[(df['Release Date'] >= 1900) & (df['Release Date'] <= 2025)]
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_content_based_recommendations(df, preferred_genres=None, content_types=None, min_rating=0, n_recommendations=10):
    """Simple content-based recommendations"""
    filtered_df = df.copy()
    
    # Filter by content type
    if content_types:
        filtered_df = filtered_df[filtered_df['Content Type'].isin(content_types)]
    
    # Filter by genres
    if preferred_genres:
        genre_filter = '|'.join(preferred_genres)
        filtered_df = filtered_df[
            filtered_df['Genres'].str.contains(genre_filter, case=False, na=False)
        ]
    
    # Filter by minimum rating
    if min_rating > 0:
        filtered_df = filtered_df[filtered_df['Imdb_Numeric'] >= min_rating]
    
    # Sort by IMDB score and return top recommendations
    recommendations = filtered_df.nlargest(n_recommendations, 'Imdb_Numeric')
    
    return recommendations

def create_similarity_matrix(df):
    """Create content similarity matrix using TF-IDF"""
    # Combine features for similarity calculation
    df['combined_features'] = (
        df['Genres'].fillna('') + ' ' +
        df['Cast'].fillna('') + ' ' +
        df['Director'].fillna('') + ' ' +
        df['Description'].fillna('')
    )
    
    # Clean text
    df['combined_features'] = df['combined_features'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower()))
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return similarity_matrix

def get_similar_content(df, title, similarity_matrix, n_recommendations=10):
    """Get content similar to a specific title"""
    try:
        # Find the index of the given title
        idx = df[df['Title'].str.contains(title, case=False, na=False)].index[0]
        
        # Get similarity scores for this content
        similarity_scores = list(enumerate(similarity_matrix[idx]))
        
        # Sort by similarity score
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar content (excluding the input title itself)
        similar_indices = [i[0] for i in similarity_scores[1:n_recommendations+1]]
        
        return df.iloc[similar_indices]
    
    except (IndexError, KeyError):
        return pd.DataFrame()

def main():
    # Header
    st.markdown('<h1 class="main-header">Netflix Content Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Content Explorer", "Get Recommendations", "Similar Content", "Analytics Dashboard"]
    )
    
    if page == "Content Explorer":
        content_explorer_page(df)
    elif page == "Get Recommendations":
        recommendations_page(df)
    elif page == "Similar Content":
        similar_content_page(df)
    elif page == "Analytics Dashboard":
        analytics_page(df)

def content_explorer_page(df):
    """Content exploration and filtering page"""
    st.header("Content Explorer")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Content", f"{len(df):,}")
    with col2:
        movies_count = len(df[df['Content Type'] == 'Movie'])
        st.metric("Movies", f"{movies_count:,}")
    with col3:
        shows_count = len(df[df['Content Type'] == 'TV Show'])
        st.metric("TV Shows", f"{shows_count:,}")
    with col4:
        avg_rating = df['Imdb_Numeric'].mean()
        st.metric("Avg IMDB Score", f"{avg_rating:.1f}/10")
    
    # Filters
    st.subheader("Filter Content")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        content_type = st.selectbox("Content Type", ["All"] + list(df['Content Type'].unique()))
    
    with col2:
        genres = df['Genres'].str.split(', ').explode().unique()
        genres = [g for g in genres if pd.notna(g) and g != 'Unknown']
        selected_genre = st.selectbox("Genre", ["All"] + sorted(genres))
    
    with col3:
        min_rating = st.slider("Minimum IMDB Rating", 0.0, 10.0, 0.0, 0.5)
    
    # Apply filters
    filtered_df = df.copy()
    
    if content_type != "All":
        filtered_df = filtered_df[filtered_df['Content Type'] == content_type]
    
    if selected_genre != "All":
        filtered_df = filtered_df[filtered_df['Genres'].str.contains(selected_genre, case=False, na=False)]
    
    if min_rating > 0:
        filtered_df = filtered_df[filtered_df['Imdb_Numeric'] >= min_rating]
    
    # Display results
    st.subheader(f"Found {len(filtered_df):,} items")
    
    if len(filtered_df) > 0:
        # Display content in cards
        for idx, row in filtered_df.head(20).iterrows():
            with st.container():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{row['Title']}</h4>
                    <p><strong>Type:</strong> {row['Content Type']} | <strong>Rating:</strong> {row['Rating']} | <strong>IMDB:</strong> {row['Imdb Score']}</p>
                    <p><strong>Genres:</strong> {row['Genres']}</p>
                    <p><strong>Cast:</strong> {str(row['Cast'])[:200]}...</p>
                    <p>{str(row['Description'])[:200]}...</p>
                </div>
                """, unsafe_allow_html=True)

def recommendations_page(df):
    """Recommendation generation page"""
    st.header("Get Personalized Recommendations")
    
    # User preferences
    st.subheader("Tell us your preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        genres = df['Genres'].str.split(', ').explode().unique()
        genres = [g for g in genres if pd.notna(g) and g != 'Unknown']
        preferred_genres = st.multiselect(
            "Preferred Genres",
            options=sorted(genres),
            default=[]
        )
    
    with col2:
        content_types = st.multiselect(
            "Content Types",
            options=df['Content Type'].unique(),
            default=['Movie', 'TV Show']
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        min_rating = st.slider("Minimum IMDB Rating", 0.0, 10.0, 6.0, 0.5)
    
    with col4:
        n_recommendations = st.slider("Number of Recommendations", 5, 50, 10)
    
    # Generate recommendations
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            
            recommendations = get_content_based_recommendations(
                df=df,
                preferred_genres=preferred_genres,
                content_types=content_types,
                min_rating=min_rating,
                n_recommendations=n_recommendations
            )
            
            if len(recommendations) > 0:
                st.subheader(f"Top {len(recommendations)} Recommendations for You")
                display_recommendations(recommendations)
            else:
                st.warning("No content found matching your criteria. Try adjusting your preferences.")

def similar_content_page(df):
    """Find similar content page"""
    st.header("Find Similar Content")
    
    st.subheader("Enter a title to find similar content")
    
    # Search for title
    search_title = st.text_input("Search for a title:", placeholder="Enter movie or TV show name...")
    
    if search_title:
        # Find matching titles
        matching_titles = df[df['Title'].str.contains(search_title, case=False, na=False)]
        
        if len(matching_titles) > 0:
            # Display matching titles
            st.subheader("Matching Titles:")
            selected_title = st.selectbox(
                "Select a title:",
                options=matching_titles['Title'].tolist()
            )
            
            if st.button("Find Similar Content", type="primary"):
                with st.spinner("Finding similar content..."):
                    # Create similarity matrix
                    similarity_matrix = create_similarity_matrix(df)
                    
                    # Get similar content
                    similar_content = get_similar_content(df, selected_title, similarity_matrix, 10)
                    
                    if len(similar_content) > 0:
                        st.subheader(f"Content Similar to '{selected_title}'")
                        display_recommendations(similar_content)
                    else:
                        st.warning("No similar content found.")
        else:
            st.warning("No titles found matching your search.")

def display_recommendations(recommendations):
    """Display recommendations in a nice format"""
    for idx, row in recommendations.iterrows():
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{row['Title']}</h4>
                    <p><strong>Type:</strong> {row['Content Type']} | <strong>Rating:</strong> {row['Rating']} | <strong>IMDB:</strong> {row['Imdb Score']}</p>
                    <p><strong>Genres:</strong> {row['Genres']}</p>
                    <p><strong>Cast:</strong> {str(row['Cast'])[:200]}...</p>
                    <p>{str(row['Description'])[:250]}...</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("IMDB Score", f"{row['Imdb_Numeric']:.1f}/10")
                st.metric("Release Year", int(row['Release Date']) if pd.notna(row['Release Date']) else 'Unknown')

def analytics_page(df):
    """Analytics and insights page"""
    st.header("Analytics Dashboard")
    
    # Genre distribution
    st.subheader("Content Distribution by Genre")
    
    genres_expanded = df['Genres'].str.split(', ').explode()
    genre_counts = genres_expanded.value_counts().head(15)
    
    fig = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation='h',
        title="Top 15 Genres",
        labels={'x': 'Number of Titles', 'y': 'Genre'},
        color=genre_counts.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Release year trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Content Releases by Year")
        yearly_releases = df['Release Date'].value_counts().sort_index()
        
        fig = px.line(
            x=yearly_releases.index,
            y=yearly_releases.values,
            title="Content Releases Over Time",
            labels={'x': 'Year', 'y': 'Number of Releases'},
            markers=True
        )
        fig.update_traces(line_color='#E50914')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Content Type Distribution")
        content_type_counts = df['Content Type'].value_counts()
        
        fig = px.pie(
            values=content_type_counts.values,
            names=content_type_counts.index,
            title="Movies vs TV Shows",
            color_discrete_sequence=['#E50914', '#221F1F']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Country analysis
    st.subheader("Top Production Countries")
    
    countries_expanded = df['Production Country'].str.split(', ').explode()
    country_counts = countries_expanded.value_counts().head(10)
    
    fig = px.bar(
        x=country_counts.index,
        y=country_counts.values,
        title="Top 10 Production Countries",
        labels={'x': 'Country', 'y': 'Number of Titles'},
        color=country_counts.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # IMDB Score distribution
    st.subheader("IMDB Score Distribution")
    
    fig = px.histogram(
        df,
        x='Imdb_Numeric',
        nbins=20,
        title="Distribution of IMDB Scores",
        labels={'x': 'IMDB Score', 'y': 'Number of Titles'},
        color_discrete_sequence=['#E50914']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top rated content
    st.subheader("Top Rated Content")
    
    top_rated = df.nlargest(10, 'Imdb_Numeric')[['Title', 'Content Type', 'Genres', 'Imdb Score', 'Release Date']]
    st.dataframe(top_rated, use_container_width=True)

if __name__ == "__main__":
    main()
