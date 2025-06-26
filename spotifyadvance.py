import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

def preprocess(df):
    features_all = ['danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    features = [f for f in features_all if f in df.columns]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    return df, scaled, features, scaler

def cluster_songs(scaled, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled)
    return clusters, kmeans


def recommend(df, song_name, features, scaler, n_recommend=5, genre_filter=None):
    if genre_filter:
        df = df[df['genre'] == genre_filter]

    song_row = df[df['name'].str.lower() == song_name.lower()]
    if song_row.empty:
        return None

    idx = song_row.index[0]
    scaled_features = scaler.transform(df[features])
    sim = cosine_similarity([scaled_features[idx]], scaled_features)
    scores = list(enumerate(sim[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n_recommend+1]
    top_indices = [i[0] for i in scores]

    cols = ['name', 'artists', 'genre', 'duration_ms', 'explicit']
    return df.iloc[top_indices][cols]

# ----------- Show Visuals -----------
def show_visuals(df):
    st.subheader("🎨 Feature Distributions")
    feature = st.selectbox("Choose a feature to plot", df.select_dtypes(include='number').columns)
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True, color='skyblue')
    st.pyplot(plt.gcf())

    st.subheader("📈 Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt.gcf())

# ----------- Cluster Explorer -----------
def explore_clusters(df, features):
    st.subheader("🌀 Cluster Feature Averages")
    cluster_means = df.groupby("Cluster")[features].mean()
    st.dataframe(cluster_means.style.background_gradient(cmap="Blues"), use_container_width=True)

    st.bar_chart(cluster_means)

    st.subheader("🎧 Songs in Cluster")
    selected_cluster = st.slider("Choose a cluster", 0, df['Cluster'].nunique() - 1, 0)
    st.dataframe(df[df['Cluster'] == selected_cluster][['name', 'artists', 'genre']].head(30))

# ----------- Main App -----------
def main():
    st.set_page_config(page_title="Spotify Smart Recommender", page_icon="🎧", layout="wide")

    st.markdown("""
        <h1 style='text-align: center; color: #1DB954;'>🎧 Smart Spotify Song Recommender</h1>
        <h4 style='text-align: center; color: gray;'>Discover music by features, genre, and similarity</h4><hr>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=100)
        st.markdown("### 👨‍💻 Developer: **Manish**")
        st.markdown("---")
        page = st.radio("📚 Navigate", ["🎵 Recommend Songs", "📊 Visual Analytics", "🌀 Explore Clusters", "📥 Dataset Preview"])

    df = load_data()
    df, scaled, features, scaler = preprocess(df)
    df["Cluster"], model = cluster_songs(scaled)

    if page == "🎵 Recommend Songs":
        st.subheader("🎵 Get Similar Songs")

        col1, col2 = st.columns(2)
        with col1:
            song = st.text_input("🔍 Enter a song name:")
        with col2:
            genre_filter = st.selectbox("🎼 Filter by Genre (Optional)", [""] + sorted(df['genre'].unique().tolist()))

        if st.button("Recommend"):
            result = recommend(df, song, features, scaler, genre_filter=genre_filter if genre_filter else None)
            if result is not None:
                st.success("Here are your song recommendations 🎶")
                st.dataframe(result, use_container_width=True)

                csv = result.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Recommendations", data=csv, file_name="recommendations.csv", mime='text/csv')
            else:
                st.warning("⚠️ Song not found. Check spelling or try without filter.")

    elif page == "📊 Visual Analytics":
        show_visuals(df)

    elif page == "🌀 Explore Clusters":
        explore_clusters(df, features)

    elif page == "📥 Dataset Preview":
        st.dataframe(df.head(50), use_container_width=True)

    st.markdown("""
        <hr>
        <center>🎧 Built with ❤️ by <b>Manish</b> | Powered by <i>Streamlit</i></center>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
