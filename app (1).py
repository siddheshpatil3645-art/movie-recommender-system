import streamlit as st
import pickle
import pandas as pd

# Set up page configuration
st.set_page_config(page_title="MovieFlix AI", page_icon="🎬", layout="centered")

# Load data safely
movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movie_dict)
similarity = pickle.load(open('similarity_matrix.pkl', 'rb'))

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
