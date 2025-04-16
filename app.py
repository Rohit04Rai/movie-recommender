import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the data
movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')

# Merge datasets
movies = movies.merge(credits, on='title')

# Keep relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# Convert stringified lists to actual lists
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert_cast(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count != 3:
            L.append(i['name'])
            count += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['crew'] = movies['crew'].apply(fetch_director)

# Process text columns
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new DataFrame with relevant columns
new = movies[['movie_id', 'title', 'tags']]
new.loc[:, 'tags'] = new['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
new.loc[:, 'tags'] = new['tags'].apply(lambda x: x.lower())

# Vectorize the tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new['tags']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie = movie.lower()
    if movie not in new['title'].str.lower().values:
        return ["Movie not found!"]
    
    index = new[new['title'].str.lower() == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [new.iloc[i[0]].title for i in movies_list]

# Streamlit UI
st.title("🎬 Movie Recommender System")
st.write("Select a movie below and get top 5 similar recommendations!")

selected_movie = st.selectbox("Search or select a movie:", new['title'].values)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.subheader("Top 5 Recommendations:")
    for movie in recommendations:
        st.write("👉", movie)
