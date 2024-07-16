
# from flask import Flask, request, render_template, jsonify
# import joblib
# import pandas as pd
# from fuzzywuzzy import process

# app = Flask(__name__)

# # Load the model
# model = joblib.load('movie_recommender_model.pkl')
# final_dataset = model['final_dataset']
# sig = model['sig']
# movies = pd.read_csv('tmdb_5000_movies.csv')

# import ast

# def parse_genres(genres_str):
#     try:
#         genres_list = ast.literal_eval(genres_str)
#         return [genre['name'] for genre in genres_list]
#     except (ValueError, SyntaxError):
#         return []

# movies['parsed_genres'] = movies['genres'].apply(parse_genres)

# def get_closest_match(input_genre, movie_genres):
#     closest_match = process.extractOne(input_genre, movie_genres)
#     return closest_match[0] if closest_match else None

# def recommend_by_genre(genre):
#     movie_genres = movies['parsed_genres'].explode().unique()
#     closest_genre = get_closest_match(genre, movie_genres)
#     if closest_genre is None:
#         return "No matching genre found."
#     genre_movies = movies[movies['parsed_genres'].apply(lambda x: closest_genre in x)]
#     genre_movies_sorted = genre_movies.sort_values(by='vote_average', ascending=False)
#     top_10_movies = genre_movies_sorted[['original_title', 'vote_average']].head(10)
#     return top_10_movies.values.tolist()

# def get_closest_match_title(input_title, movie_titles):
#     closest_match = process.extractOne(input_title, movie_titles)
#     return closest_match[0] if closest_match else None

# def CBF_get_recommendations(title):
#     movie_titles = final_dataset['original_title'].tolist()
#     closest_match = get_closest_match_title(title, movie_titles)
#     if closest_match:
#         idx = final_dataset[final_dataset['original_title'] == closest_match].index[0]
#         sig_scores = list(enumerate(sig[idx]))
#         sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
#         sig_scores = sig_scores[1:11]
#         movie_indices = [i[0] for i in sig_scores]
#         return final_dataset['original_title'].iloc[movie_indices].to_list()
#     else:
#         return "No movie found. Please check your input."

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     data = request.json
#     rec_type = data.get('type')
#     value = data.get('value')
#     if rec_type == 'genre':
#         recommendations = recommend_by_genre(value)
#     elif rec_type == 'movie':
#         recommendations = CBF_get_recommendations(value)
#     else:
#         recommendations = "Invalid type"
#     return jsonify(recommendations)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from fuzzywuzzy import process
import ast

app = Flask(__name__)

# Load the model
model = joblib.load('movie_recommender_model.pkl')
final_dataset = model['final_dataset']
sig = model['sig']
movies = pd.read_csv('tmdb_5000_movies.csv')

def parse_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        return [genre['name'] for genre in genres_list]
    except (ValueError, SyntaxError):
        return []

movies['parsed_genres'] = movies['genres'].apply(parse_genres)

def get_closest_match(input_genre, movie_genres):
    closest_match = process.extractOne(input_genre, movie_genres)
    return closest_match[0] if closest_match else None

def recommend_by_genre(genre):
    movie_genres = movies['parsed_genres'].explode().unique()
    closest_genre = get_closest_match(genre, movie_genres)
    if closest_genre is None:
        return []
    genre_movies = movies[movies['parsed_genres'].apply(lambda x: closest_genre in x)]
    genre_movies_sorted = genre_movies.sort_values(by='vote_average', ascending=False)
    top_10_movies = genre_movies_sorted.head(10)
    return top_10_movies[['original_title', 'vote_average']].to_dict(orient='records')

def get_closest_match_title(input_title, movie_titles):
    closest_match = process.extractOne(input_title, movie_titles)
    return closest_match[0] if closest_match else None

def CBF_get_recommendations(title):
    movie_titles = final_dataset['original_title'].tolist()
    closest_match = get_closest_match_title(title, movie_titles)
    if closest_match:
        idx = final_dataset[final_dataset['original_title'] == closest_match].index[0]
        sig_scores = list(enumerate(sig[idx]))
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
        sig_scores = sig_scores[1:11]
        movie_indices = [i[0] for i in sig_scores]
        recommendations = final_dataset['original_title'].iloc[movie_indices].to_list()
        return [{"title": title} for title in recommendations]
    else:
        return [{"title": "No movie found. Please check your input."}]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    rec_type = data.get('type')
    value = data.get('value')
    if rec_type == 'genre':
        recommendations = recommend_by_genre(value)
    elif rec_type == 'movie':
        recommendations = CBF_get_recommendations(value)
    else:
        recommendations = "Invalid type"
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
