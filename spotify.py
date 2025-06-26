# spotify_recommender.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df.drop_duplicates(inplace=True)
    return df

# ------------------ Preprocessing ------------------
def preprocess(df):
    features = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    return df, df_scaled, features, scaler

# ------------------ KMeans Clustering ------------------
def cluster_songs(scaled_data, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    return clusters, kmeans

# ------------------ Recommender Function ------------------
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
    return df.iloc[recommended_indices][['name', 'artists', 'genre']]

# ------------------ Visualizations ------------------
def show_visuals(df):
    st.subheader("Genre Distribution")
    genre_count = df['genre'].value_counts().head(10)
    st.bar_chart(genre_count)

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
    st.pyplot(plt.gcf())

# ------------------ Streamlit UI ------------------
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
        filtered_df = df[df['Cluster'] == cluster_num][['name', 'artists', 'genre']]
        st.dataframe(filtered_df.head(20))

if __name__ == "__main__":
    main()
