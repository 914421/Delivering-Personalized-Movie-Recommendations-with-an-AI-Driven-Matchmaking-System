import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import difflib

# Load a manageable subset of the dataset
chunk_size = 1000  # Adjust based on memory constraints
df_chunk = pd.read_csv("pradeep.csv", nrows=chunk_size)

# Basic preprocessing: clean and combine relevant features
def clean_data(x):
    return x.lower().strip().replace(" ", "") if isinstance(x, str) else ""

required_features = ["genres", "keywords", "cast", "director", "title"]
for feature in required_features:
    if feature not in df_chunk.columns:
        df_chunk[feature] = ""
    df_chunk[feature] = df_chunk[feature].fillna("").apply(clean_data)

df_chunk["combined_features"] = df_chunk["genres"] + " " + df_chunk["keywords"] + " " + df_chunk["cast"] + " " + df_chunk["director"]

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df_chunk["combined_features"])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Reset index and construct reverse mapping
df_chunk = df_chunk.reset_index()
indices = pd.Series(df_chunk.index, index=df_chunk["title"].str.lower().str.strip()).drop_duplicates()

# Function to find closest match
def get_closest_match(title):
    possible_matches = df_chunk["title"].str.lower().str.strip().tolist()
    matches = difflib.get_close_matches(title.lower().strip(), possible_matches, n=1, cutoff=0.6)
    return matches[0] if matches else None

# Recommendation function with improved matching
def get_recommendations(title, cosine_sim=cosine_sim):
    matched_title = get_closest_match(title)
    if not matched_title:
        return f"'{title}' not found in the dataset."
    
    idx = indices[matched_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # Top 10 recommendations
    movie_indices = [i[0] for i in sim_scores]
    
    return df_chunk["title"].iloc[movie_indices]

# Example usage
    if __name__ == "__main__":
        print("AI Movie Recommender System")
        user_input = input("Enter a movie title: ")
        recommendations = get_recommendations(user_input)
        print("\nTop 10 Recommendations:\n")
        if isinstance(recommendations, str):
            print(recommendations)
            else:
                for i, title in enumerate(recommendations, 1):
                    print(f"{i}. {title}")
