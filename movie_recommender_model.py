

pip install fuzzywuzzy python-Levenshtein

import pandas as pd
import numpy as np
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#

movies = pd.read_csv('tmdb_5000_movies.csv')
movies.head()

credits = pd.read_csv('tmdb_5000_credits.csv')
credits.head()

print('movies shape:', movies.shape)
print('credits shape:', credits.shape)

credits.rename(columns={'movie_id':'id'}, inplace=True)
credits.head()

final_dataset = movies.merge(credits, on='id')
final_dataset.head()

final_dataset.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'], inplace=True)
final_dataset.head()

final_dataset.info()

final_dataset.head(1)['overview']

from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern= r'\w{1,}',
            ngram_range=(1,3), stop_words='english')

# filling empty overview columns with a NaN
final_dataset['overview'] = final_dataset['overview'].fillna('')

tfv_matrix = tfv.fit_transform(final_dataset['overview'])

tfv_matrix

tfv_matrix.shape

from sklearn.metrics.pairwise import sigmoid_kernel

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

sig[0]

indices = pd.Series(final_dataset.index, index=final_dataset['original_title']).drop_duplicates()

indices

def get_closest_match(input_title, movie_titles):
    closest_match = process.extractOne(input_title, movie_titles)
    return closest_match[0] if closest_match else None

def CBF_get_recommendations(title,sig=sig):
  movie_titles = final_dataset['original_title'].tolist()
  closest_match = get_closest_match(title, movie_titles)
  if closest_match:
    idx = indices[closest_match]

    sig_scores = list(enumerate(sig[idx]))

    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    sig_scores = sig_scores[1:11]

    movie_indices = [i[0] for i in sig_scores]

    return final_dataset['original_title'].iloc[movie_indices]
  else:
    print("No movie found. Please check your input.")

CBF_get_recommendations('harry')

"""***Recommendation By Genre***"""

final_dataset.iloc[0].genres

import ast

def parse_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        return [genre['name'] for genre in genres_list]
    except (ValueError, SyntaxError):
        return []

movies['parsed_genres'] = movies['genres'].apply(parse_genres)

movies.iloc[0].parsed_genres

movies_df = movies[['original_title', 'parsed_genres', 'vote_average']]


movies_df.columns = ['Movie Title', 'Parsed Genres', 'Vote Average']

movies_df.index = movies_df.index + 1


print(movies_df.to_string(index_names=False))

from fuzzywuzzy import process
import pandas as pd

# Assuming final_dataset is your DataFrame with columns 'original_title', 'parsed_genres', and 'vote_average'

def get_closest_match(input_genre, movie_genres):
    closest_match = process.extractOne(input_genre, movie_genres)
    return closest_match[0] if closest_match else None

def recommend_by_genre(genre):
    # Get list of all parsed genres
    movie_genres = movies['parsed_genres'].explode().unique()

    # Find the closest matching genre
    closest_genre = get_closest_match(genre, movie_genres)
    if closest_genre is None:
        return "No matching genre found."

    # Filter movies by the closest matching genre
    genre_movies = movies[movies['parsed_genres'].apply(lambda x: closest_genre in x)]

    # Sort movies by vote_average in descending order
    genre_movies_sorted = genre_movies.sort_values(by='vote_average', ascending=False)

    # Select top 10 movies
    top_10_movies = genre_movies_sorted.head(10)

    # Return DataFrame with selected columns
    return top_10_movies[['original_title', 'parsed_genres', 'vote_average']]

recommend_by_genre('scifi')

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib

# Load datasets
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets
credits_df.rename(columns={'movie_id':'id'}, inplace=True)
final_dataset = pd.merge(movies_df, credits_df, on='id')

# Train the model (example)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(final_dataset['overview'].fillna(''))
sig = linear_kernel(tfidf_matrix, tfidf_matrix)

# Save the model components into a dictionary
model = {
    'tfidf': tfidf,
    'sig': sig,
    'final_dataset': final_dataset,
    'movies': movies_df  # Include the movies dataset as part of the model components
}

# Save the model as a .pkl file
joblib_file = "movie_recommender_model.pkl"
joblib.dump(model, joblib_file)

# Download the file
from google.colab import files
files.download(joblib_file)