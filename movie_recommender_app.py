import streamlit as st
import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from pathlib import Path

CSV_NAME = "imdb_top_1000.csv"
W2V_NAME = "Word2Vec_Movie_Model.model"
STOPWORDS = {"the","a","an","and","of","in","on","for","to","with","is","are","as","by","at","from","this","that","it"}

def locate_file(name):
    p = Path(name)
    if p.exists():
        return p
    icloud = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Documents" / name
    if icloud.exists():
        return icloud
    raise FileNotFoundError(f"{name} not found.")

@st.cache_data
def load_data():
    path = locate_file(CSV_NAME)
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(' ', '_')
    if "Series_Title" in df.columns:
        df = df.rename(columns={
            "Series_Title": "title",
            "Overview": "overview",
            "Genre": "genre",
            "Released_Year": "year",
            "IMDB_Rating": "rating"
        })
    df['title'] = df['title'].astype(str)
    df['overview'] = df.get('overview', "").fillna("")
    df['genre'] = df.get('genre', "").fillna("")
    df['year'] = df.get('year', "")
    df['Certificate'] = df.get('Certificate', "N/A")
    df['Runtime'] = df.get('Runtime', "N/A")
    df['Meta_score'] = df.get('Meta_score', "N/A")
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    else:
        df['rating'] = None
    df['combined'] = (df['title'].astype(str) + " " + df['genre'].astype(str) + " " + df['overview'].astype(str)).str.lower()
    return df.reset_index(drop=True)

def tokenize(text):
    words = re.findall(r"\w+", str(text).lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]

def get_doc_vector(text, model):
    tokens = tokenize(text)
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype=float)
    return np.mean(vecs, axis=0)

def build_vectors(df, w2v_model):
    vecs = []
    for t in df['combined']:
        vecs.append(get_doc_vector(t, w2v_model))
    return np.vstack(vecs)

def cosine_sim(a, M):
    a_norm = np.linalg.norm(a)
    M_norm = np.linalg.norm(M, axis=1)
    if a_norm == 0:
        return np.zeros(M.shape[0])
    return M.dot(a) / (M_norm * a_norm + 1e-9)

def recommend_by_movie(df, vecs, idx, top_n=5):
    sims = cosine_sim(vecs[idx], vecs)
    order = np.argsort(sims)[::-1]
    recs = []
    for i in order:
        if i == idx:
            continue
        recs.append((int(i), float(sims[i])))
        if len(recs) >= top_n:
            break
    return recs

def recommend_by_genre(df, genre_query, top_n=5):
    q = genre_query.strip().lower()
    subset = df[df['genre'].astype(str).str.lower().str.contains(q)]
    if subset.empty:
        return []
    if subset['rating'].notna().any():
        subset = subset.sort_values(by='rating', ascending=False)
    return [int(i) for i in subset.head(top_n).index]

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("**Owner: Navvya Chaudhary**")
st.markdown("---")

df = load_data()
w2v = Word2Vec.load(str(locate_file(W2V_NAME)))
vecs = build_vectors(df, w2v)

menu = st.sidebar.radio("Choose an option:", ["Find a Movie", "Recommend by Movie", "Recommend by Genre", "About"])

if menu == "Find a Movie":
    name = st.text_input("Enter movie name:")
    if st.button("Search"):
        result = df[df['title'].str.contains(name, case=False, na=False)]
        if not result.empty:
            for _, row in result.head(5).iterrows():
                st.subheader(row['title'])
                st.write(f"**Genre:** {row['genre']} | **Year:** {row['year']} | **Rating:** {row['rating']}")
                st.write(f"**Certificate:** {row['Certificate']} | **Runtime:** {row['Runtime']} | **Meta Score:** {row['Meta_score']}")
                st.write(row['overview'])
                st.markdown("---")
        else:
            st.warning("No movie found. Try another name.")

elif menu == "Recommend by Movie":
    movie = st.text_input("Enter a movie name to get recommendations:")
    if st.button("Recommend"):
        name_low = movie.strip().lower()
        idxs = df[df['title'].str.lower() == name_low].index
        if not idxs.empty:
            idx = idxs[0]
            recs = recommend_by_movie(df, vecs, idx)
            st.subheader(f"🎞 Recommendations similar to {df.iloc[idx]['title']}:")
            for i, score in recs:
                r = df.iloc[i]
                st.write(f"**{r['title']}** ({r['year']}) | {r['genre']}")
                st.write(f"Rating: {r['rating']} | Meta Score: {r['Meta_score']}")
                st.markdown("---")
        else:
            st.warning("Movie not found.")

elif menu == "Recommend by Genre":
    genre = st.text_input("Enter a genre (e.g., Drama, Action, Romance):")
    if st.button("Show Recommendations"):
        ids = recommend_by_genre(df, genre, top_n=5)
        if not ids:
            st.error("No movies found for that genre.")
        else:
            st.subheader(f"Top movies in genre '{genre.title()}':")
            for i in ids:
                r = df.iloc[i]
                st.write(f"🎬 **{r['title']}** ({r['year']})")
                st.write(f"Rating: {r['rating']} | Certificate: {r['Certificate']} | Runtime: {r['Runtime']} | Meta Score: {r['Meta_score']}")
                st.markdown("---")

elif menu == "About":
    st.write("This Movie Recommendation System uses a **Word2Vec semantic model** trained on movie metadata to find relationships between movies.")
    st.write("It helps users discover movies similar to their favorites or filter by genre using natural language processing techniques.")
    st.success("Developed by Navvya Chaudhary | BTech CSE (AI & ML)")
