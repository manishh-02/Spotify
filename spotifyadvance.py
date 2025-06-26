import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ----------- Load Data -----------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

# ----------- Preprocess -----------
def preprocess(df):
    features_all = ['danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    features = [f for f in features_all if f in df.columns]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    return df, scaled, features, scaler

# ----------- Cluster Songs -----------
def cluster_songs(scaled, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled)
    return clusters, kmeans

# ----------- Recommend Songs -----------
def recommend(df, song_name, features, scaler, n_recommend=5, genre_filter=None, genre_col=None):
    if genre_filter and genre_col:
        df = df[df[genre_col] == genre_filter]

    song_row = df[df['name'].str.lower() == song_name.lower()]
    if song_row.empty:
        return None

    idx = song_row.index[0]
    scaled_features = scaler.transform(df[features])
    sim = cosine_similarity([scaled_features[idx]], scaled_features)
    scores = list(enumerate(sim[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n_recommend+1]
    top_indices = [i[0] for i in scores]

    cols = ['name', 'artists']
    if genre_col: cols.append(genre_col)
    if 'duration_ms' in df.columns: cols.append('duration_ms')
    if 'explicit' in df.columns: cols.append('explicit')

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
def explore_clusters(df, features, genre_col):
    st.subheader("🌀 Cluster Feature Averages")
    cluster_means = df.groupby("Cluster")[features].mean()
    st.dataframe(cluster_means.style.background_gradient(cmap="Blues"), use_container_width=True)
    st.bar_chart(cluster_means)

    st.subheader("🎧 Songs in Selected Cluster")
    selected_cluster = st.slider("Choose a cluster", 0, df['Cluster'].nunique() - 1, 0)
    cols = ['name', 'artists']
    if genre_col: cols.append(genre_col)
    st.dataframe(df[df['Cluster'] == selected_cluster][cols].head(30), use_container_width=True)

# ----------- Main App -----------
def main():
    st.set_page_config(page_title="Spotify Smart Recommender", page_icon="🎧", layout="wide")

    # -------- Developer Name at the TOP --------
    st.markdown("""
        <h2 style='text-align: center; color: #333;'>👨‍💻 Developed by <span style="color:#1DB954;">Manish</span></h2>
    """, unsafe_allow_html=True)

    # -------- App Title --------
    st.markdown("""
        <h1 style='text-align: center; color: #1DB954;'>🎧 Spotify Music Recommender</h1>
        <h4 style='text-align: center; color: gray;'>Discover music by similarity, genre & mood</h4><hr>
    """, unsafe_allow_html=True)

    # -------- Sidebar --------
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=100)
        st.markdown("### 👨‍💻 App by: **Manish**")
        st.markdown("---")
        page = st.radio("📚 Choose Option", ["🎵 Recommend Songs", "📊 Visual Analytics", "🌀 Explore Clusters", "📥 Dataset Preview"])

    # -------- Load Data --------
    df = load_data()
    df, scaled, features, scaler = preprocess(df)
    df["Cluster"], model = cluster_songs(scaled)

    # -------- Determine genre column --------
    genre_col = None
    for col in ['genre', 'genres']:
        if col in df.columns:
            genre_col = col
            break

    # -------- Pages --------
    if page == "🎵 Recommend Songs":
        st.subheader("🎵 Find Similar Songs")

        col1, col2 = st.columns(2)
        with col1:
            song = st.text_input("🔍 Enter a Song Name:")
        with col2:
            if genre_col:
                genre_filter = st.selectbox("🎼 Filter by Genre (Optional)", [""] + sorted(df[genre_col].dropna().unique().tolist()))
            else:
                st.warning("⚠️ No 'genre' or 'genres' column found.")
                genre_filter = None

        if st.button("🎯 Recommend"):
            result = recommend(df, song, features, scaler, genre_filter=genre_filter if genre_filter else None, genre_col=genre_col)
            if result is not None:
                st.success("🎧 Recommended Songs:")
                st.dataframe(result, use_container_width=True)
                # Download
                csv = result.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download CSV", data=csv, file_name="recommended_songs.csv", mime='text/csv')
            else:
                st.error("❌ Song not found. Please check the name and try again.")

    elif page == "📊 Visual Analytics":
        show_visuals(df)

    elif page == "🌀 Explore Clusters":
        explore_clusters(df, features, genre_col)

    elif page == "📥 Dataset Preview":
        st.subheader("🔎 Sample of Dataset")
        st.dataframe(df.head(50), use_container_width=True)

    # -------- Footer --------
    st.markdown("""
        <hr>
        <center>
        Built with ❤️ by <b>Manish</b> | Powered by <i>Streamlit</i>
        </center>
    """, unsafe_allow_html=True)

# ----------- Run App -----------
if __name__ == "__main__":
    main()
