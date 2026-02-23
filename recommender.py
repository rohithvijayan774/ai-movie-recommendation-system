import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load saved objects
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("movies.pkl", "rb") as f:
    df = pickle.load(f)

# Create TF-IDF matrix again
tfidf_matrix = tfidf.transform(df['combined'])

def recommend(movie_title, top_n=5):
    movie_title = movie_title.lower()

    matches = df[df['title'].str.lower() == movie_title]

    if matches.empty:
        return []

    idx = matches.index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    )[0]

    scores = list(enumerate(similarity_scores))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_movies = scores[1:top_n+1]
    indices = [i[0] for i in top_movies]

    return df['title'].iloc[indices].tolist()
