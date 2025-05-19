import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import difflib

st.title("AI Movie Recommender System")

# Load data
@st.cache_data
def load_data():
    chunk_size = 1000  # Adjust as needed
    df = pd.read_csv("pradeep.csv", nrows=chunk_size)
    
    def clean_data(x):
        return x.lower().strip().replace(" ", "") if isinstance(x, str) else ""

    required_features = ["genres", "keywords", "cast", "director", "title"]
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = ""
        df[feature] = df[feature].fillna("").apply(clean_data)

    df["combined_features"] = df["genres"] + " " + df["keywords"] + " " + df["cast"] + " " + df["director"]
    
    return df

df_chunk = load_data()

# TF-IDF Vectorization and similarity
@st.cache_resource
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df["title"].str.lower().str.strip()).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = compute_similarity(df_chunk)

# Title matching
def get_closest_match(title):
    possible_matches = df_chunk["title"].str.lower().str.strip().tolist()
    matches = difflib.get_close_matches(title.lower().strip(), possible_matches, n=1, cutoff=0.6)
    return matches[0] if matches else None

# Recommendation function
def get_recommendations(title):
    matched_title = get_closest_match(title)
    if not matched_title:
        return None, f"'{title}' not found in the dataset."
    
    idx = indices[matched_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = df_chunk["title"].iloc[movie_indices].tolist()
    
    return recommendations, None

# UI
user_input = st.text_input("Enter a movie title:")
if user_input:
    results, error = get_recommendations(user_input)
    if error:
        st.error(error)
    else:
        st.subheader("Top 10 Recommendations:")
        for i, title in enumerate(results, 1):
            st.write(f"{i}. {title}")
