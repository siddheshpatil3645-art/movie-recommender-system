import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up page configuration
st.set_page_config(page_title="MovieFlix AI", page_icon="🎬", layout="centered")

# Load data safely
@st.cache_resource
def load_data():
    movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies_df = pd.DataFrame(movie_dict)
    
    # Calculate similarity matrix dynamically to avoid large file upload limits
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['tags'])
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return movies_df, sim_matrix

movies, similarity = load_data()

st.title("🎬 MovieFlix AI")
st.markdown("Select a movie you like, and we'll suggest 5 similar ones!")

# Dropdown selection
selected_movie = st.selectbox(
    "Select Movie",
    movies['title'].values
)

# Recommendation function
def recommend(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    distances = similarity[idx]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# Button to show recommendations
if st.button("Recommend", type="primary"):
    recommendations = recommend(selected_movie)
    st.markdown("### 🍿 You might also like:")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.info(recommendations[i])
