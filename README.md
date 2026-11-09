# Netflix Content Recommendation System

A Streamlit app for browsing a Netflix catalog and getting title recommendations. It has two recommendation modes: a filter-and-rank mode (pick genres, content types and a minimum IMDb score, get the highest rated matches) and a "similar titles" mode that uses TF-IDF and cosine similarity on each title's genres, cast, director and description. There is also a small analytics page with charts of the catalog.

## Data

`netflixData.csv` is a snapshot of the Netflix catalog with 5,967 titles (3,867 movies and 2,100 TV shows). Columns: Show Id, Title, Description, Director, Genres, Cast, Production Country, Release Date (year), Rating, Duration, Imdb Score (e.g. `6.6/10`), Content Type, Date Added. Release years go up to 2021, and 608 titles have no IMDb score.

There is no user or viewing data in the dataset, so the app does not do collaborative filtering. Everything is based on title metadata.

## How it works

Preprocessing (`load_data` in `streamlit_app.py`):
- missing genres, cast, director, country and rating are filled with placeholders
- the IMDb score is parsed to a number; missing scores are filled with the median
- rows without a valid release year are dropped, leaving 5,964 titles

Pages:
- **Content Explorer**: filter by type, genre and minimum IMDb score.
- **Get Recommendations**: filters by the selected genres and types, then returns the top N titles by IMDb score.
- **Similar Content**: builds a TF-IDF matrix (1,000 features, unigrams and bigrams, English stop words removed) over genres + cast + director + description, then returns the 10 titles with the highest cosine similarity to the chosen title.
- **Analytics Dashboard**: top genres, releases per year, movies vs TV shows, top production countries, IMDb score distribution and the top rated titles.

There is no offline evaluation of the recommendations, so no accuracy numbers are reported here.

## How to run

Requires Python 3.10+.

```bash
git clone https://github.com/KaanDRoid/Netflix-Content-Recommendation-System.git
cd Netflix-Content-Recommendation-System
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at http://localhost:8501. Run it from the repo root, since the CSV is loaded with a relative path.

## Repo layout

```
streamlit_app.py   # the Streamlit app (data loading, recommendations, charts)
netflixData.csv    # dataset
requirements.txt
LICENSE            # MIT
```

## Possible next steps

- add real user ratings so a collaborative filtering model can be trained and evaluated
- add a simple offline check for the similarity results (e.g. genre overlap of the top 10)
