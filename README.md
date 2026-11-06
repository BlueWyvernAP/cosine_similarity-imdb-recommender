# 🎬 IMDB Movie Recommender

Een interactieve film-aanbeveler gebouwd met **Streamlit**, aangedreven door **TF-IDF** en **cosine similarity**.

## ✨ Features

- 🔍 **Zoek een film** en krijg directe aanbevelingen
- 📊 **Interactieve similarity grafiek** — zie hoe films zich tot elkaar verhouden in een 2D ruimte (t-SNE)
- 🎭 **Browse op genre** via de sidebar
- 🔥 **Populairste films** als startpagina
- 🖼️ Filmposters via TMDB CDN

## 🤖 Hoe werkt het?

| Stap | Techniek |
|------|----------|
| Tekstrepresentatie | TF-IDF op genres + keywords + cast + plot |
| Aanbevelingen | Cosine similarity (on-demand per zoekopdracht) |
| Visualisatie | TruncatedSVD → t-SNE → 2D Plotly scatter |

## 📁 Bestandsstructuur

```
imdb_recommender/
├── app.py               # Hoofdapplicatie
├── requirements.txt     # Python dependencies
└── README.md            # Dit bestand
```

## 🚀 Lokaal draaien

1. **Clone de repo**
   ```bash
   git clone https://github.com/jouw-gebruikersnaam/imdb-recommender.git
   cd imdb-recommender
   ```

2. **Installeer dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Zet het dataset-pad goed** in `app.py`:
   ```python
   DATASET_PATH = r"pad/naar/jouw/movies_dataset.csv"
   ```

4. **Start de app**
   ```bash
   streamlit run app.py
   ```

5. Open je browser op **http://localhost:8501**

## 📊 Dataset

De app gebruikt `movies_dataset.csv` (`;`-gescheiden) met de volgende kolommen:
`id`, `title`, `genres`, `overview`, `popularity`, `release_date`, `vote_average`, `vote_count`, `credits`, `keywords`, `poster_path`

> ⚠️ De dataset is te groot voor GitHub (366 MB). Download hem apart en pas `DATASET_PATH` aan in `app.py`.

## 🛠️ Tech stack

- [Streamlit](https://streamlit.io/) — web UI
- [scikit-learn](https://scikit-learn.org/) — TF-IDF, cosine similarity, SVD, t-SNE
- [Plotly](https://plotly.com/) — interactieve grafieken
- [pandas](https://pandas.pydata.org/) — data verwerking
