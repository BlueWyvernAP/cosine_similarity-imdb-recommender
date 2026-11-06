import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
import re

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .movie-card {
    background: #1c1f26;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
    border: 1px solid #2e3340;
  }
  .movie-title { font-size: 15px; font-weight: bold; color: #fff; margin: 6px 0 2px; }
  .movie-meta  { font-size: 12px; color: #aaa; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 600; margin: 2px 2px 0;
  }
  .badge-genre { background: #e50914; color: #fff; }
  .badge-score { background: #f5c518; color: #000; }
  .hero { text-align: center; padding: 30px 0 10px; }
  .hero h1 { font-size: 3em; color: #e50914; }
  .hero p  { color: #ccc; font-size: 1.1em; }
  img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
DATASET_PATH    = r"C:\Users\arjen\Downloads\VSC\movies_dataset.csv"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w300"
PLACEHOLDER_IMG = "https://via.placeholder.com/200x300?text=No+Poster"
NUM_RECS        = 10
MIN_VOTES       = 50

# ─── Data loading & preprocessing ───────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Dataset laden… (eerste keer ~20 sec)")
def load_data():
    df = pd.read_csv(
        DATASET_PATH,
        sep=";",
        on_bad_lines="skip",
        encoding="latin-1",
        low_memory=False,
    )
    keep = ["id", "title", "genres", "overview", "popularity",
            "release_date", "vote_average", "vote_count",
            "credits", "keywords", "poster_path"]
    df = df[[c for c in keep if c in df.columns]].copy()

    for col in ["genres", "overview", "credits", "keywords", "title"]:
        df[col] = df[col].fillna("").astype(str)

    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0)
    df["vote_count"]   = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0)
    df["popularity"]   = pd.to_numeric(df["popularity"],   errors="coerce").fillna(0)

    df = df[df["vote_count"] >= MIN_VOTES].copy()
    df["year"] = df["release_date"].str.extract(r"(\d{4})").fillna("?")
    df["soup"] = (
        df["genres"].str.replace("-", " ") + " " +
        df["keywords"].str.replace("-", " ") + " " +
        df["credits"].str.replace("-", " ").str[:200] + " " +
        df["overview"].str[:300]
    )
    df = df.reset_index(drop=True)
    return df


@st.cache_resource(show_spinner="🔢 TF-IDF berekenen… (eenmalig ~30 sec)")
def build_tfidf(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=20_000, ngram_range=(1, 2))
    return tfidf.fit_transform(df["soup"])


# ─── Recommendation engine ───────────────────────────────────────────────────
def get_similar(movie_title: str, df: pd.DataFrame, tfidf_matrix, n: int = 50):
    """Returns (source_row, similar_df_with_scores, source_pos) or (None, empty, -1)."""
    mask = df["title"].str.lower().str.contains(movie_title.lower(), na=False, regex=False)
    matches = df[mask]
    if matches.empty:
        return None, pd.DataFrame(), -1

    idx = matches["vote_count"].idxmax()
    pos = df.index.get_loc(idx)

    movie_vec = tfidf_matrix[pos]
    scores = cosine_similarity(movie_vec, tfidf_matrix).flatten()
    scores[pos] = -1
    top_positions = scores.argsort()[::-1][:n]

    result = df.iloc[top_positions].copy()
    result["similarity"] = scores[top_positions]
    return df.loc[idx], result, pos


# ─── t-SNE visualisation ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="📐 t-SNE berekenen… (~15 sec)", ttl=600)
def compute_tsne(_tfidf_matrix, indices: tuple, perplexity: int = 30):
    """Reduce TF-IDF vectors for the given indices to 2D with t-SNE."""
    sub = _tfidf_matrix[list(indices)]
    # First reduce to 50 dims with SVD (much faster for t-SNE)
    n_components = min(50, sub.shape[1] - 1, sub.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(sub)
    perp = min(perplexity, len(indices) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=500)
    coords = tsne.fit_transform(reduced)
    return coords


# ─── UI helpers ─────────────────────────────────────────────────────────────
def poster_url(path):
    p = str(path).strip()
    if not p or p == "nan":
        return PLACEHOLDER_IMG
    return p if p.startswith("http") else TMDB_IMAGE_BASE + p


def star_rating(score):
    filled = int(round(score / 2))
    return "⭐" * filled + "☆" * (5 - filled)


def genre_badges(genres_str):
    genres = [g.strip() for g in re.split(r"[-,|]", genres_str) if g.strip()]
    return " ".join(f'<span class="badge badge-genre">{g}</span>' for g in genres[:4])


def first_genre(genres_str):
    parts = [g.strip() for g in re.split(r"[-,|]", genres_str) if g.strip()]
    return parts[0] if parts else "Unknown"


def render_movie_card(row, col):
    with col:
        st.image(poster_url(row.get("poster_path", "")), use_container_width=True)
        score = float(row.get("vote_average", 0))
        overview = str(row.get("overview", ""))
        short = overview[:110] + "…" if len(overview) > 110 else overview
        st.markdown(f"""
        <div class="movie-card">
          <div class="movie-title">{row['title']} ({row.get('year','?')})</div>
          <div class="movie-meta">
            <span class="badge badge-score">⭐ {score:.1f}</span>
            {genre_badges(str(row.get('genres','')))}
          </div>
          <div class="movie-meta" style="margin-top:6px;font-size:11px;">{short}</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Similarity scatter plot ─────────────────────────────────────────────────
def render_similarity_graph(source_movie, similar_df, source_pos, tfidf_matrix, n_display):
    sim_top = similar_df.head(n_display).copy()
    all_positions = [source_pos] + [
        int(similar_df.index.get_loc(i)) if i in similar_df.index else source_pos
        for i in sim_top.index
    ]
    # Build a clean position list from DataFrame positional indices
    source_label = source_movie["title"]

    # Collect positions: source + top similar
    positions = [source_pos]
    for idx in sim_top.index:
        try:
            positions.append(similar_df.index.get_loc(idx))
        except Exception:
            pass

    # We need actual positions in the FULL tfidf_matrix
    # sim_top.index are the original df integer indices
    full_positions = [source_pos] + list(sim_top.index)

    with st.spinner("📐 t-SNE berekenen voor visualisatie… (~15 sec)"):
        coords = compute_tsne(tfidf_matrix, tuple(full_positions))

    # Build plot dataframe
    plot_rows = [source_movie] + [sim_top.loc[i] for i in sim_top.index]
    plot_df = pd.DataFrame(plot_rows).reset_index(drop=True)
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    plot_df["is_source"] = [True] + [False] * len(sim_top)
    plot_df["similarity"] = [1.0] + list(sim_top["similarity"].values)
    plot_df["genre_main"] = plot_df["genres"].apply(first_genre)
    plot_df["vote_average"] = pd.to_numeric(plot_df["vote_average"], errors="coerce").fillna(0)
    plot_df["label"] = plot_df["title"] + " (" + plot_df["year"].astype(str) + ")"
    plot_df["sim_pct"] = (plot_df["similarity"] * 100).round(1).astype(str) + "%"

    # ── Plotly figure ────────────────────────────────────────────────────────
    genres_in_plot = plot_df["genre_main"].unique()
    color_palette = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    color_map = {g: color_palette[i % len(color_palette)] for i, g in enumerate(genres_in_plot)}

    fig = go.Figure()

    # Draw similarity lines from source to each similar movie
    sx, sy = float(plot_df.loc[0, "x"]), float(plot_df.loc[0, "y"])
    for _, row in plot_df.iloc[1:].iterrows():
        sim = float(row["similarity"])
        fig.add_trace(go.Scatter(
            x=[sx, row["x"]], y=[sy, row["y"]],
            mode="lines",
            line=dict(color=f"rgba(255,255,255,{sim * 0.4:.2f})", width=sim * 2),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Draw similar movies per genre
    for genre in genres_in_plot:
        gdf = plot_df[(plot_df["genre_main"] == genre) & (~plot_df["is_source"])]
        if gdf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=gdf["x"], y=gdf["y"],
            mode="markers+text",
            name=genre,
            marker=dict(
                size=gdf["vote_average"] * 1.8 + 6,
                color=color_map[genre],
                opacity=0.85,
                line=dict(width=1, color="white"),
            ),
            text=gdf["title"].str[:18],
            textposition="top center",
            textfont=dict(size=9, color="white"),
            customdata=np.stack([
                gdf["title"], gdf["year"].astype(str),
                gdf["vote_average"].round(1).astype(str),
                gdf["genres"], gdf["sim_pct"]
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "⭐ %{customdata[2]} | 🎭 %{customdata[3]}<br>"
                "🔗 Similarity: %{customdata[4]}<extra></extra>"
            ),
        ))

    # Draw source movie (highlighted)
    fig.add_trace(go.Scatter(
        x=[sx], y=[sy],
        mode="markers+text",
        name=f"🎯 {source_label}",
        marker=dict(size=28, color="#e50914", symbol="star",
                    line=dict(width=2, color="white")),
        text=[source_label[:20]],
        textposition="bottom center",
        textfont=dict(size=11, color="#e50914", family="Arial Black"),
        hovertemplate=f"<b>{source_label}</b><br>⭐ {float(source_movie.get('vote_average',0)):.1f}<br>🎭 {source_movie.get('genres','')}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"🗺️ Similarity map — films gelijkend op <b>{source_label}</b>",
            font=dict(size=18, color="white"),
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        legend=dict(
            bgcolor="rgba(28,31,38,0.9)",
            bordercolor="#2e3340",
            borderwidth=1,
            font=dict(size=11),
        ),
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend info
    st.markdown("""
    > **📖 Hoe lezen?**  
    > - 🔴 **Ster** = de gezochte film  
    > - **Kleur** = genre  
    > - **Grootte** van de stip = IMDB-score  
    > - **Lijndikte** = hoe sterk de cosine similarity is  
    > - Films die **dichter bij de ster** staan lijken meer op jouw film  
    > Hover over een film voor details
    """)


# ─── Main app ────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="hero">
      <h1>🎬 Movie Recommender</h1>
      <p>Vind films die lijken op jouw favoriete film — aangedreven door TF-IDF &amp; cosine similarity</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    tfidf_matrix = build_tfidf(df)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg", width=120)
        st.markdown("### ⚙️ Instellingen")
        n_recs = st.slider("Aantal aanbevelingen", 5, 20, NUM_RECS)
        n_graph = st.slider("Films in grafiek", 20, 100, 40)
        st.markdown("---")
        st.markdown(f"**📊 Dataset:** `{len(df):,}` films geladen")
        st.markdown("**🔍 Methode:** TF-IDF + Cosine Similarity")
        st.markdown("**📐 Visualisatie:** TruncatedSVD → t-SNE → 2D")
        st.markdown(f"**🎚️ Filter:** ≥ {MIN_VOTES} stemmen")
        st.markdown("---")

        st.markdown("### 🎭 Browse op genre")
        all_genres = sorted(set(
            g.strip()
            for gs in df["genres"].dropna()
            for g in re.split(r"[-,|]", gs) if g.strip()
        ))
        selected_genre = st.selectbox("Genre", ["— alle —"] + all_genres)

    # ── Zoekveld (gedeeld tussen tabs) ───────────────────────────────────────
    query = st.text_input(
        "🔍 Zoek een film (bijv. *Inception*, *Avengers*, *Fast X*)",
        placeholder="Typ een filmtitel…",
        key="search"
    )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🍿 Aanbevelingen", "📊 Similarity Grafiek"])

    # ════════════════════════════════════════
    # TAB 1 — Aanbevelingen
    # ════════════════════════════════════════
    with tab1:
        if query.strip():
            with st.spinner("🔎 Aanbevelingen berekenen…"):
                source_movie, similar_df, source_pos = get_similar(
                    query.strip(), df, tfidf_matrix, n=max(n_recs, n_graph)
                )

            if source_movie is None:
                st.error(f"❌ Geen film gevonden voor **{query}**. Probeer een andere titel.")
            else:
                st.markdown("---")
                st.markdown(f"### 🎯 Op basis van: **{source_movie['title']}** ({source_movie.get('year','?')})")
                src_col1, src_col2 = st.columns([1, 4])
                with src_col1:
                    st.image(poster_url(source_movie.get("poster_path", "")), width=130)
                with src_col2:
                    score = float(source_movie.get("vote_average", 0))
                    st.markdown(f"**⭐ IMDB score:** {score:.1f} / 10  {star_rating(score)}")
                    st.markdown(f"**🎭 Genres:** {source_movie.get('genres','?').replace('-',' · ')}")
                    st.markdown(f"**🏷️ Keywords:** {source_movie.get('keywords','?').replace('-', ' · ')[:100]}")
                    st.markdown(f"**🎬 Cast:** {str(source_movie.get('credits',''))[:120]}")
                    st.markdown(f"**📖 Plot:** {str(source_movie.get('overview',''))[:300]}")

                st.markdown("---")
                top_recs = similar_df.head(n_recs)
                st.markdown(f"### 🍿 Top {len(top_recs)} aanbevelingen")
                cols = st.columns(5)
                for i, (_, row) in enumerate(top_recs.iterrows()):
                    render_movie_card(row, cols[i % 5])

        elif selected_genre != "— alle —":
            st.markdown(f"### 🎭 Top films in genre: **{selected_genre}**")
            genre_df = df[df["genres"].str.contains(selected_genre, na=False, case=False)]
            top = genre_df.nlargest(20, "vote_count")
            cols = st.columns(5)
            for i, (_, row) in enumerate(top.iterrows()):
                render_movie_card(row, cols[i % 5])

        else:
            st.markdown("### 🔥 Populairste films")
            top_popular = df.nlargest(15, "popularity")
            cols = st.columns(5)
            for i, (_, row) in enumerate(top_popular.iterrows()):
                render_movie_card(row, cols[i % 5])
            st.markdown("---")
            st.info("💡 Typ een filmtitel hierboven om gepersonaliseerde aanbevelingen te krijgen!")

    # ════════════════════════════════════════
    # TAB 2 — Similarity Grafiek
    # ════════════════════════════════════════
    with tab2:
        if not query.strip():
            st.markdown("### 📊 Similarity Grafiek")
            st.info("🔍 Zoek eerst een film in het zoekveld hierboven om de grafiek te zien.")
            st.markdown("""
            **Wat zie je hier?**
            - Een 2D kaart van films op basis van inhoudelijke gelijkenis
            - Berekend met **TF-IDF → TruncatedSVD → t-SNE**
            - Films die dicht bij elkaar staan hebben vergelijkbare genres, cast en plot
            - De rode ster is jouw gezochte film
            """)
        else:
            with st.spinner("🔎 Films ophalen…"):
                source_movie, similar_df, source_pos = get_similar(
                    query.strip(), df, tfidf_matrix, n=n_graph
                )

            if source_movie is None:
                st.error(f"❌ Geen film gevonden voor **{query}**.")
            else:
                st.markdown(f"### 📊 Similarity map voor: **{source_movie['title']}**")
                st.caption(f"Toont de {n_graph} meest gelijkende films in een 2D ruimte via t-SNE")
                render_similarity_graph(
                    source_movie, similar_df, source_pos, tfidf_matrix, n_graph
                )


if __name__ == "__main__":
    main()
