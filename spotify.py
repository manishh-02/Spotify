# spotify_recommender.py

import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ----------- Load and unzip data -----------
@st.cache_data
def load_data():
    if not os.path.exists("data.csv"):
        if os.path.exists("data.zip"):
            with zipfile.ZipFile("data.zip", 'r') as zip_ref:
                zip_ref.extractall()
        else:
            raise FileNotFoundError("❌ data.zip not found in project directory.")
    return pd.read_csv("data.csv")

# ----------- Preprocessing -----------
def preprocess(df):
    features_all = ['danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    features = [f for f in features_all if f in df.columns]

    if not features:
        raise ValueError("❌ None of the expected features found in dataset.")

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    return df, df_scaled, features, scaler

# ----------- Clustering -----------
def cluster_songs(scaled_data, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    return clusters, kmeans

# ----------- Recommendation -----------
def recommend(df, song_name, features, scaler, n_recommend=5):
    song_row = df[df['name'].str.lower() == song_name.lower()]
    if song_row.empty:
        return None

    song_index = song_row.index[0]
    scaled_features = scaler.transform(df[features])
    sim = cosine_similarity([scaled_features[song_index]], scaled_features)
    sim_scores = list(enumerate(sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommend+1]
    recommended_indices = [i[0] for i in sim_scores]

    if 'genre' in df.columns:
        cols = ['name', 'artists', 'genre']
    elif 'genres' in df.columns:
        cols = ['name', 'artists', 'genres']
    else:
        cols = ['name', 'artists']

    return df.iloc[recommended_indices][cols]

# ----------- Visualizations -----------
def show_visuals(df):
    st.subheader("🎼 Genre Distribution")
    genre_col = 'genre' if 'genre' in df.columns else 'genres' if 'genres' in df.columns else None

    if genre_col:
        genre_count = df[genre_col].value_counts().head(10)
        st.bar_chart(genre_count)
    else:
        st.warning("No genre/genres column found.")

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
    st.pyplot(plt.gcf())

# ----------- Main App -----------
def main():
    st.title("🎧 Spotify Music Recommender & Cluster Analyzer")

    df = load_data()
    df, scaled_data, features, scaler = preprocess(df)
    clusters, kmeans_model = cluster_songs(scaled_data)
    df['Cluster'] = clusters

    menu = st.sidebar.selectbox("Choose Option", ["Recommender", "Visualizations", "Cluster Explorer"])

    if menu == "Recommender":
        st.subheader("🎵 Get Similar Songs")
        song = st.text_input("Enter a song name:")
        if st.button("Recommend"):
            result = recommend(df, song, features, scaler)
            if result is not None:
                st.write("### Recommended Songs:")
                st.dataframe(result)
            else:
                st.warning("Song not found. Please check spelling.")

    elif menu == "Visualizations":
        show_visuals(df)

    elif menu == "Cluster Explorer":
        st.subheader("Explore Clusters")
        cluster_num = st.slider("Select cluster number", 0, df['Cluster'].nunique() - 1, 0)
        genre_col = 'genre' if 'genre' in df.columns else 'genres' if 'genres' in df.columns else None
        cols = ['name', 'artists']
        if genre_col:
            cols.append(genre_col)
        filtered_df = df[df['Cluster'] == cluster_num][cols]
        st.dataframe(filtered_df.head(20))

if __name__ == "__main__":
    main()

