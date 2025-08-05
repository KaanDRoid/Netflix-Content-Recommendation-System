import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_genre_distribution_chart(df):
    """Create interactive genre distribution chart"""
    
    # Expand genres and count
    genres_expanded = df['Genres'].str.split(', ').explode()
    genre_counts = genres_expanded.value_counts().head(15)
    
    fig = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation='h',
        title="Content Distribution by Genre",
        labels={'x': 'Number of Titles', 'y': 'Genre'},
        color=genre_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_release_trend_chart(df):
    """Create release trends over time"""
    
    yearly_releases = df['Release Date'].value_counts().sort_index()
    
    fig = px.line(
        x=yearly_releases.index,
        y=yearly_releases.values,
        title="Content Releases by Year",
        labels={'x': 'Year', 'y': 'Number of Releases'},
        markers=True
    )
    
    fig.update_traces(line_color='#E50914')
    fig.update_layout(height=400)
    
    return fig

def create_country_distribution_chart(df):
    """Create country distribution pie chart"""
    
    countries_expanded = df['Production Country'].str.split(', ').explode()
    country_counts = countries_expanded.value_counts().head(10)
    
    fig = px.pie(
        values=country_counts.values,
        names=country_counts.index,
        title="Top 10 Production Countries"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    
    return fig

def create_rating_distribution_chart(df):
    """Create content rating distribution"""
    
    rating_counts = df['Rating'].value_counts()
    
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        title="Content by Rating Category",
        labels={'x': 'Rating', 'y': 'Number of Titles'},
        color=rating_counts.values,
        color_continuous_scale='Plasma'
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def create_imdb_score_distribution(df):
    """Create IMDB score distribution histogram"""
    
    # Convert IMDB scores to numeric
    imdb_numeric = pd.to_numeric(df['Imdb Score'].str.replace('/10', ''), errors='coerce')
    
    fig = px.histogram(
        x=imdb_numeric,
        nbins=20,
        title="IMDB Score Distribution",
        labels={'x': 'IMDB Score', 'y': 'Number of Titles'}
    )
    
    fig.update_traces(marker_color='#E50914')
    fig.update_layout(height=400)
    
    return fig

def create_content_type_comparison(df):
    """Create content type comparison chart"""
    
    content_stats = df.groupby('Content Type').agg({
        'Title': 'count',
        'Imdb Score': lambda x: pd.to_numeric(x.str.replace('/10', ''), errors='coerce').mean()
    }).round(2)
    
    content_stats.columns = ['Count', 'Avg_IMDB_Score']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Content Count', 'Average IMDB Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Count chart
    fig.add_trace(
        go.Bar(
            x=content_stats.index,
            y=content_stats['Count'],
            name='Count',
            marker_color='#E50914'
        ),
        row=1, col=1
    )
    
    # Average IMDB score chart
    fig.add_trace(
        go.Bar(
            x=content_stats.index,
            y=content_stats['Avg_IMDB_Score'],
            name='Avg IMDB Score',
            marker_color='#221F1F'
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, title_text="Movies vs TV Shows Comparison")
    
    return fig

def create_duration_analysis(df):
    """Create duration analysis charts"""
    
    # Movies duration
    movies = df[df['Content Type'] == 'Movie'].copy()
    movies['Duration_Minutes'] = movies['Duration'].str.extract(r'(\d+) min').astype(float)
    
    # TV Shows seasons
    tv_shows = df[df['Content Type'] == 'TV Show'].copy()
    tv_shows['Seasons'] = tv_shows['Duration'].str.extract(r'(\d+) Season').astype(float)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Movie Duration Distribution', 'TV Show Seasons Distribution')
    )
    
    # Movie duration histogram
    fig.add_trace(
        go.Histogram(
            x=movies['Duration_Minutes'],
            name='Movie Duration',
            marker_color='#E50914',
            nbinsx=20
        ),
        row=1, col=1
    )
    
    # TV show seasons histogram
    fig.add_trace(
        go.Histogram(
            x=tv_shows['Seasons'],
            name='TV Show Seasons',
            marker_color='#221F1F',
            nbinsx=10
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, title_text="Content Duration Analysis")
    fig.update_xaxes(title_text="Duration (Minutes)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Seasons", row=1, col=2)
    fig.update_yaxes(title_text="Number of Titles", row=1, col=1)
    fig.update_yaxes(title_text="Number of Shows", row=1, col=2)
    
    return fig

def create_recommendation_performance_chart(performance_data):
    """Create recommendation model performance comparison"""
    
    algorithms = list(performance_data.keys())
    precision_scores = [performance_data[alg]['precision_at_k'] for alg in algorithms]
    recall_scores = [performance_data[alg]['recall_at_k'] for alg in algorithms]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Precision@K',
        x=algorithms,
        y=precision_scores,
        marker_color='#E50914'
    ))
    
    fig.add_trace(go.Bar(
        name='Recall@K',
        x=algorithms,
        y=recall_scores,
        marker_color='#221F1F'
    ))
    
    fig.update_layout(
        title="Recommendation Algorithm Performance Comparison",
        xaxis_title="Algorithm",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    return fig

def create_genre_network_graph(df, top_n=20):
    """Create genre co-occurrence network graph"""
    
    # Get genre combinations
    genre_combinations = []
    
    for genres_str in df['Genres'].dropna():
        genres = [g.strip() for g in genres_str.split(',')]
        if len(genres) > 1:
            for i in range(len(genres)):
                for j in range(i + 1, len(genres)):
                    genre_combinations.append((genres[i], genres[j]))
    
    # Count combinations
    from collections import Counter
    combo_counts = Counter(genre_combinations)
    
    # Get top combinations
    top_combos = combo_counts.most_common(top_n)
    
    # Create network data
    nodes = set()
    edges = []
    
    for (genre1, genre2), count in top_combos:
        nodes.add(genre1)
        nodes.add(genre2)
        edges.append({
            'source': genre1,
            'target': genre2,
            'weight': count
        })
    
    # Convert to format suitable for visualization
    nodes_list = list(nodes)
    node_indices = {node: i for i, node in enumerate(nodes_list)}
    
    edge_x = []
    edge_y = []
    edge_info = []
    
    # Simple circular layout
    import math
    n_nodes = len(nodes_list)
    
    node_x = []
    node_y = []
    
    for i, node in enumerate(nodes_list):
        angle = 2 * math.pi * i / n_nodes
        x = math.cos(angle)
        y = math.sin(angle)
        node_x.append(x)
        node_y.append(y)
    
    # Create edges
    for edge in edges:
        x0, y0 = node_x[node_indices[edge['source']]], node_y[node_indices[edge['source']]]
        x1, y1 = node_x[node_indices[edge['target']]], node_y[node_indices[edge['target']]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=nodes_list,
        textposition="middle center",
        marker=dict(
            size=20,
            color='#E50914',
            line=dict(width=2, color='#221F1F')
        )
    ))
    
    fig.update_layout(
        title="Genre Co-occurrence Network",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[
            dict(
                text="Genres that frequently appear together",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="#888", size=12)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig
