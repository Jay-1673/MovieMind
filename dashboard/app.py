import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import sys
import plotly.express as px
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


# ============================================================
# TMDB POSTER API
# ============================================================
# TMDB is used only for movie artwork/posters in the dashboard.
# The IMDb dataset remains the source of the ML data and features.
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")


@st.cache_data(show_spinner=False)
def get_movie_poster(title, year=None):
    """Return a TMDB poster URL for a movie title and optional year."""

    if not TMDB_API_KEY:
        return None

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": str(title),
        "include_adult": False
    }

    if year is not None:
        try:
            params["year"] = int(float(year))
        except (TypeError, ValueError):
            pass

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        if response.status_code != 200:
            return None

        results = response.json().get("results", [])

        if not results:
            return None

        for result in results:
            poster_path = result.get("poster_path")

            if poster_path:
                return (
                    "https://image.tmdb.org/t/p/w500"
                    + poster_path
                )

        return None

    except requests.RequestException:
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_tmdb_movie_data(title, year=None):
    """Return TMDB details, watch providers and trailer information."""

    if not TMDB_API_KEY:
        return None

    search_url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": str(title),
        "include_adult": False
    }

    if year is not None:
        try:
            params["year"] = int(float(year))
        except (TypeError, ValueError):
            pass

    try:
        search_response = requests.get(
            search_url,
            params=params,
            timeout=10
        )

        if search_response.status_code != 200:
            return None

        results = search_response.json().get("results", [])

        if not results:
            return None

        # Prefer an exact title match when TMDB provides one.
        target_title = str(title).strip().lower()
        match = next(
            (
                item for item in results
                if str(item.get("title", "")).strip().lower() == target_title
            ),
            results[0]
        )

        movie_id = match.get("id")

        if not movie_id:
            return None

        details_response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={
                "api_key": TMDB_API_KEY,
                "append_to_response": "videos"
            },
            timeout=10
        )

        providers_response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers",
            params={"api_key": TMDB_API_KEY},
            timeout=10
        )

        details = (
            details_response.json()
            if details_response.status_code == 200
            else match
        )

        provider_data = (
            providers_response.json().get("results", {}).get("IN", {})
            if providers_response.status_code == 200
            else {}
        )

        return {
            "id": movie_id,
            "title": details.get("title", match.get("title", title)),
            "overview": details.get("overview", ""),
            "tagline": details.get("tagline", ""),
            "runtime": details.get("runtime"),
            "release_date": details.get("release_date", ""),
            "genres": [g.get("name") for g in details.get("genres", [])],
            "homepage": details.get("homepage", ""),
            "tmdb_url": provider_data.get(
                "link",
                f"https://www.themoviedb.org/movie/{movie_id}"
            ),
            "streaming": provider_data.get("flatrate", []),
            "free": provider_data.get("free", []),
            "ads": provider_data.get("ads", []),
            "rent": provider_data.get("rent", []),
            "buy": provider_data.get("buy", []),
            "videos": details.get("videos", {}).get("results", [])
        }

    except (requests.RequestException, ValueError, TypeError):
        return None
    except Exception:
        return None


def provider_names(providers):
    """Return unique provider names from a TMDB provider list."""
    names = []
    for provider in providers or []:
        name = provider.get("provider_name")
        if name and name not in names:
            names.append(name)
    return names


# ============================================================
# PREMIUM CINEMATIC UI
# ============================================================

st.markdown(
"""
<style>

/* =========================================================
   GLOBAL
========================================================= */

.stApp {

    background:
        radial-gradient(
            circle at 5% 0%,
            rgba(99,102,241,0.18),
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 10%,
            rgba(168,85,247,0.16),
            transparent 25%
        ),
        linear-gradient(
            180deg,
            #06070d 0%,
            #0a0d16 50%,
            #06070d 100%
        );

    color: #f5f7ff;
}


.block-container {

    max-width: 1450px;

    padding-top: 1.5rem;
    padding-bottom: 5rem;
}


/* =========================================================
   HEADINGS
========================================================= */

h1 {

    font-size: 3.2rem !important;
    font-weight: 900 !important;
    letter-spacing: -1.5px;
}


h2 {

    font-size: 2rem !important;
    font-weight: 850 !important;
}


h3 {

    font-weight: 800 !important;
}


/* =========================================================
   HERO
========================================================= */

.hero-box {

    position: relative;

    padding: 55px 50px;

    margin-bottom: 35px;

    border-radius: 28px;

    background:
        linear-gradient(
            135deg,
            rgba(27,31,58,0.98),
            rgba(10,13,24,0.98)
        );

    border:
        1px solid rgba(255,255,255,0.09);

    box-shadow:
        0 25px 70px rgba(0,0,0,0.45);

    overflow: hidden;
}


.hero-box::before {

    content: "";

    position: absolute;

    width: 400px;
    height: 400px;

    right: -120px;
    top: -180px;

    background:
        radial-gradient(
            circle,
            rgba(124,58,237,0.35),
            transparent 70%
        );

    pointer-events: none;
}


.hero-box::after {

    content: "";

    position: absolute;

    width: 350px;
    height: 350px;

    left: -180px;
    bottom: -220px;

    background:
        radial-gradient(
            circle,
            rgba(79,70,229,0.25),
            transparent 70%
        );

    pointer-events: none;
}


.hero-title {

    position: relative;

    font-size: 46px;

    font-weight: 900;

    letter-spacing: -1.5px;

    margin-bottom: 12px;

    z-index: 2;
}


.hero-subtitle {

    position: relative;

    max-width: 850px;

    font-size: 18px;

    line-height: 1.7;

    color: rgba(255,255,255,0.68);

    z-index: 2;
}


/* =========================================================
   METRIC CARDS
========================================================= */

[data-testid="stMetric"] {

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.06),
            rgba(255,255,255,0.025)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius: 20px;

    padding: 22px;

    min-height: 110px;

    box-shadow:
        0 10px 35px rgba(0,0,0,0.20);

    transition:
        transform 0.25s ease,
        border 0.25s ease,
        box-shadow 0.25s ease;
}


[data-testid="stMetric"]:hover {

    transform:
        translateY(-5px);

    border:
        1px solid rgba(129,140,248,0.45);

    box-shadow:
        0 15px 40px rgba(79,70,229,0.18);
}


/* =========================================================
   BUTTONS
========================================================= */

.stButton > button {

    width: 100%;

    min-height: 52px;

    border-radius: 14px;

    border:
        1px solid rgba(255,255,255,0.12);

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        );

    color: white;

    font-size: 16px;

    font-weight: 800;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}


.stButton > button:hover {

    transform:
        translateY(-3px);

    box-shadow:
        0 12px 35px rgba(99,102,241,0.40);
}


/* =========================================================
   INPUT
========================================================= */

div[data-baseweb="input"] {

    border-radius: 14px;

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid rgba(255,255,255,0.08);
}


div[data-baseweb="input"]:focus-within {

    border:
        1px solid rgba(129,140,248,0.65);

    box-shadow:
        0 0 0 2px rgba(99,102,241,0.12);
}


/* =========================================================
   TABS
========================================================= */

button[data-baseweb="tab"] {

    font-size: 15px;

    font-weight: 750;

    padding:
        14px 22px;
}


button[data-baseweb="tab"]:hover {

    color: #a5b4fc;
}


/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #10121d,
            #080a11
        );

    border-right:
        1px solid rgba(255,255,255,0.07);
}


/* =========================================================
   SIDEBAR CARDS
========================================================= */

.sidebar-card {

    padding: 14px;

    margin:
        8px 0;

    border-radius: 12px;

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid rgba(255,255,255,0.06);
}


/* =========================================================
   RECOMMENDATION CARD
========================================================= */

.movie-card {

    padding: 18px 20px;

    margin:
        10px 0;

    border-radius: 18px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.055),
            rgba(255,255,255,0.018)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 8px 25px rgba(0,0,0,0.20);

    transition:
        transform 0.25s ease,
        border 0.25s ease;
}


.movie-card:hover {

    transform:
        translateY(-4px);

    border:
        1px solid rgba(129,140,248,0.45);
}


/* =========================================================
   TMDB POSTER
========================================================= */

.tmdb-poster {
    width: 100%;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 12px 35px rgba(0,0,0,0.30);
}

.poster-label {
    text-align: center;
    margin-top: 8px;
    margin-bottom: 12px;
    font-size: 11px;
    opacity: 0.45;
}


/* =========================================================
   SECTION CARD
========================================================= */

.section-card {

    padding: 25px;

    margin:
        15px 0;

    border-radius: 20px;

    background:
        rgba(255,255,255,0.025);

    border:
        1px solid rgba(255,255,255,0.07);
}


/* =========================================================
   MODEL BADGE
========================================================= */

.model-badge {

    display: inline-block;

    padding:
        6px 12px;

    margin-right: 6px;

    border-radius: 20px;

    background:
        rgba(99,102,241,0.16);

    border:
        1px solid rgba(129,140,248,0.25);

    font-size: 12px;

    font-weight: 700;
}


/* =========================================================
   DATAFRAMES
========================================================= */

[data-testid="stDataFrame"] {

    border-radius: 18px;

    overflow: hidden;

    border:
        1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 10px 30px rgba(0,0,0,0.20);
}


/* =========================================================
   ALERTS
========================================================= */

[data-testid="stAlert"] {

    border-radius: 16px;

    border:
        1px solid rgba(255,255,255,0.08);
}


/* =========================================================
   DIVIDERS
========================================================= */

hr {

    border-color:
        rgba(255,255,255,0.07);
}


/* =========================================================
   SCROLLBAR
========================================================= */

::-webkit-scrollbar {

    width: 8px;
}


::-webkit-scrollbar-track {

    background:
        #070910;
}


::-webkit-scrollbar-thumb {

    background:
        #34384a;

    border-radius:
        10px;
}


::-webkit-scrollbar-thumb:hover {

    background:
        #4f46e5;
}

</style>
""",
unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    metadata = pd.read_csv(
        os.path.join(
            MODEL_DIR,
            "knn_movie_metadata.csv"
        )
    )

    features = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "movie_features.csv"
        )
    )

    clusters = pd.read_csv(
        os.path.join(
            MODEL_DIR,
            "movie_clusters.csv"
        )
    )

    full_metadata = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "movie_metadata.csv"
        )
    )

    return (
        metadata,
        features,
        clusters,
        full_metadata
    )


# ============================================================
# LOAD KNN
# ============================================================

@st.cache_resource
def load_knn():

    path = os.path.join(
        MODEL_DIR,
        "knn_movie_model.pkl"
    )

    with open(
        path,
        "rb"
    ) as file:

        return pickle.load(file)


# ============================================================
# LOAD ML MODELS
# ============================================================

@st.cache_resource
def load_prediction_models():

    with open(
        os.path.join(
            MODEL_DIR,
            "decision_tree_model.pkl"
        ),
        "rb"
    ) as file:

        decision_tree = pickle.load(file)


    with open(
        os.path.join(
            MODEL_DIR,
            "random_forest_model.pkl"
        ),
        "rb"
    ) as file:

        random_forest = pickle.load(file)


    with open(
        os.path.join(
            MODEL_DIR,
            "linear_regression_model.pkl"
        ),
        "rb"
    ) as file:

        linear_regression = pickle.load(file)


    with open(
        os.path.join(
            MODEL_DIR,
            "naive_bayes_model.pkl"
        ),
        "rb"
    ) as file:

        naive_bayes = pickle.load(file)


    return (
        decision_tree,
        random_forest,
        linear_regression,
        naive_bayes
    )


# ============================================================
# PREPARE AI SYSTEM
# ============================================================

@st.cache_resource
def prepare_system():

    (
        metadata,
        features,
        clusters,
        full_metadata
    ) = load_data()

    knn = load_knn()

    models = load_prediction_models()


    reliable_mask = (
        full_metadata[
            "reliable_rating"
        ] == 1
    )


    features_knn = features[
        reliable_mask
    ].reset_index(
        drop=True
    )


    knn_feature_columns = list(
        features_knn.columns
    )


    X_knn = features_knn[
        knn_feature_columns
    ].values.astype(
        np.float32
    )


    metadata = metadata.merge(
        clusters[
            [
                "tconst",
                "cluster"
            ]
        ],
        on="tconst",
        how="left"
    )


    return (
        metadata,
        features_knn,
        X_knn,
        knn,
        models
    )


# ============================================================
# FIND MOVIE
# ============================================================

def find_movie(movie_title):

    (
        metadata,
        features_knn,
        X_knn,
        knn,
        models
    ) = prepare_system()


    exact_matches = metadata[
        metadata[
            "primaryTitle"
        ]
        .str.lower()
        .str.strip()
        ==
        movie_title.lower().strip()
    ]


    if not exact_matches.empty:

        return exact_matches.iloc[0]


    partial_matches = metadata[
        metadata[
            "primaryTitle"
        ]
        .str.lower()
        .str.contains(
            movie_title.lower(),
            na=False
        )
    ]


    if partial_matches.empty:

        return None


    return partial_matches.iloc[0]


# ============================================================
# HYBRID RECOMMENDATION ENGINE
# ============================================================

def get_recommendations(
    movie_title,
    number_of_recommendations=10
):

    (
        metadata,
        features_knn,
        X_knn,
        knn,
        models
    ) = prepare_system()


    selected_movie = find_movie(
        movie_title
    )


    if selected_movie is None:

        return None, None


    selected_tconst = (
        selected_movie["tconst"]
    )


    matching_indices = metadata.index[
        metadata[
            "tconst"
        ]
        ==
        selected_tconst
    ]


    if len(matching_indices) == 0:

        return None, None


    movie_index = matching_indices[0]


    movie_vector = X_knn[
        movie_index
    ].reshape(
        1,
        -1
    )


    search_count = min(
        max(
            number_of_recommendations * 5,
            50
        ),
        len(X_knn)
    )


    distances, indices = (
        knn.kneighbors(
            movie_vector,
            n_neighbors=search_count
        )
    )


    recommendations = metadata.iloc[
        indices[0]
    ].copy()


    recommendations[
        "knn_similarity"
    ] = (
        1 - distances[0]
    ) * 100


    recommendations = recommendations[
        recommendations[
            "tconst"
        ]
        != selected_tconst
    ]


    recommendations[
        "rating_score"
    ] = (
        recommendations[
            "averageRating"
        ]
        / 10
    ) * 100


    log_votes = np.log1p(
        recommendations[
            "numVotes"
        ]
    )


    max_votes = log_votes.max()


    if max_votes > 0:

        recommendations[
            "popularity_score"
        ] = (
            log_votes
            / max_votes
        ) * 100

    else:

        recommendations[
            "popularity_score"
        ] = 0


    selected_cluster = (
        selected_movie[
            "cluster"
        ]
    )


    recommendations[
        "cluster_score"
    ] = np.where(
        recommendations[
            "cluster"
        ]
        ==
        selected_cluster,
        100,
        0
    )


    recommendations[
        "hybrid_score"
    ] = (

        recommendations[
            "knn_similarity"
        ] * 0.60

        +

        recommendations[
            "rating_score"
        ] * 0.20

        +

        recommendations[
            "popularity_score"
        ] * 0.10

        +

        recommendations[
            "cluster_score"
        ] * 0.10
    )


    recommendations = (
        recommendations
        .sort_values(
            "hybrid_score",
            ascending=False
        )
        .head(
            number_of_recommendations
        )
    )


    return (
        selected_movie,
        recommendations
    )


# ============================================================
# AI PREDICTIONS
# ============================================================

def get_predictions(movie):

    (
        metadata,
        features_knn,
        X_knn,
        knn,
        models
    ) = prepare_system()


    (
        decision_tree,
        random_forest,
        linear_regression,
        naive_bayes
    ) = models


    movie_index = metadata.index[
        metadata[
            "tconst"
        ]
        ==
        movie[
            "tconst"
        ]
    ][0]


    full_vector = features_knn.iloc[
        movie_index
    ]


    # --------------------------------------------------------
    # Decision Tree
    # --------------------------------------------------------

    dt_features = (
        decision_tree["features"]
    )


    X_dt = full_vector[
        dt_features
    ].values.astype(
        np.float32
    ).reshape(
        1,
        -1
    )


    dt_prediction = (
        decision_tree["model"]
        .predict(
            X_dt
        )[0]
    )


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    rf_features = (
        random_forest["features"]
    )


    X_rf = full_vector[
        rf_features
    ].values.astype(
        np.float32
    ).reshape(
        1,
        -1
    )


    rf_prediction = (
        random_forest["model"]
        .predict(
            X_rf
        )[0]
    )


    rf_probability = (
        random_forest["model"]
        .predict_proba(
            X_rf
        )[0]
    )


    if 1 in random_forest[
        "model"
    ].classes_:

        high_rating_index = list(
            random_forest[
                "model"
            ].classes_
        ).index(1)


        high_rating_probability = (
            rf_probability[
                high_rating_index
            ] * 100
        )

    else:

        high_rating_probability = 0


    # --------------------------------------------------------
    # Linear Regression
    # --------------------------------------------------------

    lr_features = (
        linear_regression["features"]
    )


    X_lr = full_vector[
        lr_features
    ].values.astype(
        np.float32
    ).reshape(
        1,
        -1
    )


    predicted_rating = (
        linear_regression["model"]
        .predict(
            X_lr
        )[0]
    )


    predicted_rating = float(
        np.clip(
            predicted_rating,
            1,
            10
        )
    )


    # --------------------------------------------------------
    # Naive Bayes
    # --------------------------------------------------------

    nb_features = (
        naive_bayes["features"]
    )


    X_nb = full_vector[
        nb_features
    ].values.astype(
        np.float32
    ).reshape(
        1,
        -1
    )


    nb_prediction = (
        naive_bayes["model"]
        .predict(
            X_nb
        )[0]
    )


    return {

        "decision_tree": int(
            dt_prediction
        ),

        "random_forest": int(
            rf_prediction
        ),

        "rf_probability": float(
            high_rating_probability
        ),

        "predicted_rating":
            predicted_rating,

        "naive_bayes":
            nb_prediction
    }


# ============================================================
# HERO
# ============================================================

st.markdown(
"""
<div class="hero-box">

<div class="hero-title">
🎬 AI Movie Intelligence
</div>

<div class="hero-subtitle">
Intelligent movie discovery powered by Machine Learning,
Artificial Intelligence and the official IMDb dataset.
</div>

<div style="margin-top:22px;">

<span class="model-badge">
🤖 6 AI Algorithms
</span>

<span class="model-badge">
🎯 Hybrid Recommendation
</span>

<span class="model-badge">
⭐ IMDb Dataset
</span>

</div>

</div>
""",
unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
"""
<div style="
text-align:center;
padding:15px 5px 25px 5px;
">

<div style="font-size:48px;">
🎬
</div>

<div style="
font-size:23px;
font-weight:900;
">
MovieMind
</div>

<div style="
opacity:0.55;
font-size:13px;
margin-top:5px;
">
AI Recommendation Platform
</div>

</div>
""",
unsafe_allow_html=True
)


st.sidebar.divider()


st.sidebar.markdown(
"### 🤖 AI ENGINE"
)


st.sidebar.markdown(
"""
<div class="sidebar-card">
<b>🔍 KNN</b><br>
<span style="opacity:0.6;">Movie similarity</span>
</div>

<div class="sidebar-card">
<b>🧩 K-Means</b><br>
<span style="opacity:0.6;">Movie clustering</span>
</div>

<div class="sidebar-card">
<b>🌳 Decision Tree</b><br>
<span style="opacity:0.6;">Rating classification</span>
</div>

<div class="sidebar-card">
<b>🌲 Random Forest</b><br>
<span style="opacity:0.6;">Advanced classification</span>
</div>

<div class="sidebar-card">
<b>📈 Linear Regression</b><br>
<span style="opacity:0.6;">Rating prediction</span>
</div>

<div class="sidebar-card">
<b>🧠 Naive Bayes</b><br>
<span style="opacity:0.6;">Rating categories</span>
</div>
""",
unsafe_allow_html=True
)


st.sidebar.divider()


st.sidebar.markdown(
"### 🧠 HYBRID ENGINE"
)


st.sidebar.success(
"KNN + Rating + Popularity + K-Means"
)


st.sidebar.divider()


st.sidebar.caption(
"Powered by Official IMDb Dataset"
)

st.sidebar.caption(
"AI Movie Recommendation System"
)


# ============================================================
# SEARCH SECTION
# ============================================================

st.markdown(
"""
<h2>🎯 Discover Your Next Movie</h2>

<p style="opacity:0.65;font-size:16px;">
Enter a movie and let our Hybrid AI engine find movies
with similar characteristics, ratings and audience popularity.
</p>
""",
unsafe_allow_html=True
)


movie_name = st.text_input(
    "Enter a movie name",
    placeholder="Example: The Dark Knight"
)


if st.button(
    "🔍 Get AI Recommendations",
    use_container_width=True
):

    if not movie_name.strip():

        st.warning(
            "Please enter a movie name."
        )

    else:

        with st.spinner(
            "🤖 AI is analyzing the movie..."
        ):

            selected_movie, recommendations = (
                get_recommendations(
                    movie_name,
                    10
                )
            )


        if selected_movie is None:

            st.error(
                f"❌ Movie '{movie_name}' was not found."
            )

        else:

            # =================================================
            # SELECTED MOVIE
            # =================================================

            st.divider()

            st.subheader(
                "🎬 Selected Movie"
            )


            # TMDB poster is presentation-only and does not affect ML data.
            selected_poster = get_movie_poster(
                selected_movie["primaryTitle"],
                selected_movie["startYear"]
            )

            if selected_poster:
                poster_col, _ = st.columns([1, 3])

                with poster_col:
                    st.image(
                        selected_poster,
                        caption="Movie Poster • TMDB",
                        use_container_width=True
                    )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "🎬 Movie",
                    selected_movie[
                        "primaryTitle"
                    ]
                )


            with col2:

                st.metric(
                    "⭐ IMDb Rating",
                    f"{selected_movie['averageRating']:.1f}/10"
                )


            with col3:

                st.metric(
                    "📅 Release Year",
                    str(
                        selected_movie[
                            "startYear"
                        ]
                    )
                )


            with col4:

                st.metric(
                    "🧩 AI Cluster",
                    int(
                        selected_movie[
                            "cluster"
                        ]
                    )
                )


            st.markdown(
                f"""
                <div class="section-card">

                <b>🎭 Genres</b><br>
                <span style="opacity:0.7;">
                {selected_movie['genres']}
                </span>

                <br><br>

                <b>🗳️ IMDb Audience Votes</b><br>
                <span style="opacity:0.7;">
                {selected_movie['numVotes']:,}
                </span>

                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # MOVIE DETAILS + WHERE TO WATCH
            # =================================================

            st.divider()

            st.subheader("🍿 Movie Details & Where to Watch")
            st.caption(
                "Movie information and legal viewing availability are provided by TMDB/JustWatch. "
                "Your IMDb-based AI recommendation model remains unchanged."
            )

            with st.spinner("Finding movie details and legal watch options..."):
                tmdb_movie = get_tmdb_movie_data(
                    selected_movie["primaryTitle"],
                    selected_movie["startYear"]
                )

            if tmdb_movie:
                detail_col, watch_col = st.columns([1.35, 1])

                with detail_col:
                    overview = tmdb_movie.get("overview") or "No synopsis is available from TMDB."
                    tagline = tmdb_movie.get("tagline")
                    runtime = tmdb_movie.get("runtime")
                    genres = ", ".join(tmdb_movie.get("genres", [])) or str(selected_movie["genres"])

                    st.markdown("### 🎬 About the Movie")

                    if tagline:
                        st.markdown(f"*{tagline}*")

                    st.write(overview)

                    info_a, info_b, info_c = st.columns(3)
                    with info_a:
                        st.metric("Runtime", f"{runtime} min" if runtime else "N/A")
                    with info_b:
                        st.metric("Year", str(selected_movie["startYear"]))
                    with info_c:
                        st.metric("IMDb", f"{selected_movie['averageRating']:.1f}/10")

                    st.markdown(f"**Genres:** {genres}")

                    trailer = next(
                        (
                            video for video in tmdb_movie.get("videos", [])
                            if video.get("site") == "YouTube"
                            and video.get("type") == "Trailer"
                            and video.get("key")
                        ),
                        None
                    )

                    if trailer:
                        st.video(f"https://www.youtube.com/watch?v={trailer['key']}")

                with watch_col:
                    st.markdown("### 📺 Where to Watch in India")

                    streaming = provider_names(tmdb_movie.get("streaming"))
                    free = provider_names(tmdb_movie.get("free"))
                    ads = provider_names(tmdb_movie.get("ads"))
                    rent = provider_names(tmdb_movie.get("rent"))
                    buy = provider_names(tmdb_movie.get("buy"))

                    if streaming:
                        st.markdown("**▶ Streaming**")
                        st.write(" • ".join(streaming))

                    if free:
                        st.markdown("**🆓 Free**")
                        st.write(" • ".join(free))

                    if ads:
                        st.markdown("**📺 Free with Ads**")
                        st.write(" • ".join(ads))

                    if rent:
                        st.markdown("**💳 Rent**")
                        st.write(" • ".join(rent))

                    if buy:
                        st.markdown("**🛒 Buy**")
                        st.write(" • ".join(buy))

                    if any([streaming, free, ads, rent, buy]):
                        st.link_button(
                            "▶ Open Watch Options",
                            tmdb_movie["tmdb_url"],
                            use_container_width=True
                        )
                        st.caption(
                            "Availability can vary by location and can change over time. "
                            "The button opens the official TMDB watch-options page."
                        )
                    else:
                        st.warning(
                            "No India watch-provider data is currently available for this movie."
                        )
                        st.link_button(
                            "🔎 Check TMDB Watch Options",
                            tmdb_movie["tmdb_url"],
                            use_container_width=True
                        )

                    if tmdb_movie.get("homepage"):
                        st.link_button(
                            "🌐 Official Movie Website",
                            tmdb_movie["homepage"],
                            use_container_width=True
                        )

                    st.markdown(
                        "<div class='poster-label'>Watch-provider data powered by JustWatch via TMDB</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.info(
                    "TMDB details are temporarily unavailable. You can still use the IMDb-based "
                    "recommendation and AI prediction features."
                )


            # =================================================
            # AI MODEL PREDICTIONS
            # =================================================

            st.divider()

            st.subheader(
                "🧠 AI Model Predictions"
            )


            with st.spinner(
                "Running machine learning models..."
            ):

                predictions = get_predictions(
                    selected_movie
                )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "📈 Predicted Rating",
                    f"{predictions['predicted_rating']:.2f}/10"
                )


            with col2:

                st.metric(
                    "🌲 High Rating Probability",
                    f"{predictions['rf_probability']:.1f}%"
                )


            with col3:

                dt_result = (
                    "Highly Rated"
                    if predictions[
                        "decision_tree"
                    ] == 1
                    else "Not Highly Rated"
                )


                st.metric(
                    "🌳 Decision Tree",
                    dt_result
                )


            with col4:

                st.metric(
                    "🧠 Naive Bayes",
                    str(
                        predictions[
                            "naive_bayes"
                        ]
                    )
                )


            st.info(
                f"""
                **🤖 AI Analysis**

                Linear Regression predicts an IMDb rating of
                **{predictions['predicted_rating']:.2f}/10**.

                Random Forest estimates a
                **{predictions['rf_probability']:.1f}%**
                probability that the movie belongs to the
                highly-rated category.

                Decision Tree classification:
                **{dt_result}**

                Naive Bayes classification:
                **{predictions['naive_bayes']}**
                """
            )


            # =================================================
            # AI RECOMMENDATIONS
            # =================================================

            st.divider()

            st.subheader(
                "🤖 AI Recommended Movies"
            )

            st.caption(
                "Recommendations are generated using KNN similarity, "
                "IMDb rating, popularity and K-Means clustering."
            )


            recommendation_columns = st.columns(5)

            for rank, (_, movie) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                score = float(
                    movie[
                        "hybrid_score"
                    ]
                )

                with recommendation_columns[(rank - 1) % 5]:

                    poster_url = get_movie_poster(
                        movie["primaryTitle"],
                        movie["startYear"]
                    )

                    if poster_url:
                        st.image(
                            poster_url,
                            caption=f"#{rank} • TMDB Poster",
                            use_container_width=True
                        )
                    else:
                        st.markdown(
                            """
                            <div style="
                                height:280px;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                background:rgba(255,255,255,0.04);
                                border-radius:16px;
                                border:1px solid rgba(255,255,255,0.08);
                                text-align:center;
                                opacity:0.6;
                                font-size:14px;
                                margin-bottom:12px;
                            ">
                                🎬<br>Poster unavailable
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Original recommendation card content is retained.
                    st.markdown(
                        f"""
                        <div class="movie-card">

                        <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                        ">

                        <div>

                        <span style="
                        font-size:22px;
                        font-weight:900;
                        ">
                        #{rank}
                        </span>

                        &nbsp;&nbsp;

                        <span style="
                        font-size:21px;
                        font-weight:800;
                        ">
                        🎬 {movie['primaryTitle']}
                        </span>

                        </div>

                        <div style="
                        font-size:20px;
                        font-weight:900;
                        ">
                        {score:.1f}
                        </div>

                        </div>

                        <div style="
                        margin-top:10px;
                        opacity:0.65;
                        ">
                        📅 {movie['startYear']}
                        &nbsp;&nbsp; • &nbsp;&nbsp;
                        🎭 {movie['genres']}
                        </div>

                        <div style="
                        margin-top:12px;
                        ">

                        <span class="model-badge">
                        ⭐ {movie['averageRating']:.1f}
                        </span>

                        <span class="model-badge">
                        🤖 AI Score {score:.1f}
                        </span>

                        <span class="model-badge">
                        🗳️ {int(movie['numVotes']):,} votes
                        </span>

                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if rank % 5 == 0 and rank < len(recommendations):
                    recommendation_columns = st.columns(5)

# ============================================================
# SYSTEM OVERVIEW
# ============================================================

st.markdown(
"""
<h2>📊 AI System Overview</h2>
<p style="opacity:0.65;">
A multi-model Artificial Intelligence system designed
for intelligent movie discovery and analysis.
</p>
""",
unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🎬 Movies",
        "310K+"
    )


with col2:

    st.metric(
        "🤖 AI Algorithms",
        "6"
    )


with col3:

    st.metric(
        "⭐ Dataset",
        "IMDb"
    )


with col4:

    st.metric(
        "🧠 Engine",
        "Hybrid AI"
    )


# ============================================================
# DATASET ANALYTICS
# ============================================================

st.divider()

st.header(
    "📊 Dataset Analytics Dashboard"
)

st.markdown(
"""
Explore the characteristics and patterns of the
**real IMDb movie dataset** used by the AI system.
"""
)


# ============================================================
# LOAD ANALYTICS DATA
# ============================================================

@st.cache_data
def load_analytics_data():

    file_path = os.path.join(
        DATA_DIR,
        "movies_processed.csv"
    )

    analytics_data = pd.read_csv(
        file_path
    )


    if "decade" not in analytics_data.columns:

        analytics_data["decade"] = (
            pd.to_numeric(
                analytics_data[
                    "startYear"
                ],
                errors="coerce"
            )
            // 10
            * 10
        )


    if "popularity_score" not in analytics_data.columns:

        analytics_data[
            "popularity_score"
        ] = (

            analytics_data[
                "averageRating"
            ]

            *

            np.log1p(
                analytics_data[
                    "numVotes"
                ]
            )
        )


    analytics_data[
        "startYear"
    ] = pd.to_numeric(
        analytics_data[
            "startYear"
        ],
        errors="coerce"
    )


    analytics_data[
        "averageRating"
    ] = pd.to_numeric(
        analytics_data[
            "averageRating"
        ],
        errors="coerce"
    )


    analytics_data[
        "numVotes"
    ] = pd.to_numeric(
        analytics_data[
            "numVotes"
        ],
        errors="coerce"
    )


    analytics_data[
        "runtimeMinutes"
    ] = pd.to_numeric(
        analytics_data[
            "runtimeMinutes"
        ],
        errors="coerce"
    )


    return analytics_data


analytics_data = load_analytics_data()


# ============================================================
# BASIC STATISTICS
# ============================================================

total_movies = len(
    analytics_data
)


average_rating = (
    analytics_data[
        "averageRating"
    ].mean()
)


total_votes = (
    analytics_data[
        "numVotes"
    ].sum()
)


minimum_year = (
    analytics_data[
        "startYear"
    ].min()
)


maximum_year = (
    analytics_data[
        "startYear"
    ].max()
)


# ============================================================
# DATASET OVERVIEW CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🎬 Total Movies",
        f"{total_movies:,}"
    )


with col2:

    st.metric(
        "⭐ Average Rating",
        f"{average_rating:.2f}"
    )


with col3:

    st.metric(
        "🗳️ Total IMDb Votes",
        f"{total_votes:,.0f}"
    )


with col4:

    st.metric(
        "📅 Year Range",
        f"{int(minimum_year)} - {int(maximum_year)}"
    )


# ============================================================
# DATASET TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎭 Genres",
        "⭐ Ratings",
        "📅 Time Analysis",
        "🔥 Popularity"
    ]
)


# ============================================================
# TAB 1 — GENRES
# ============================================================

with tab1:

    st.subheader(
        "🎭 Movie Genre Distribution"
    )


    genre_data = analytics_data[
        [
            "genres"
        ]
    ].dropna().copy()


    genre_data[
        "genres"
    ] = genre_data[
        "genres"
    ].str.split(",")


    genre_data = genre_data.explode(
        "genres"
    )


    genre_data[
        "genres"
    ] = genre_data[
        "genres"
    ].str.strip()


    genre_counts = (
        genre_data[
            "genres"
        ]
        .value_counts()
        .head(15)
    )


    col1, col2 = st.columns(
        [1.5, 1]
    )


    with col1:

        genre_chart = pd.DataFrame(
            {
                "Genre": genre_counts.index,
                "Movies": genre_counts.values
            }
        )


        fig_genre = px.bar(
            genre_chart,
            x="Movies",
            y="Genre",
            orientation="h",
            title="Top 15 Movie Genres"
        )


        fig_genre.update_layout(
            height=550,
            template="plotly_dark"
        )


        st.plotly_chart(
            fig_genre,
            use_container_width=True
        )


    with col2:

        st.markdown(
            "### 🏆 Top Genres"
        )


        genre_table = pd.DataFrame(
            {
                "Genre":
                    genre_counts.index,

                "Movies":
                    genre_counts.values
            }
        )


        st.dataframe(
            genre_table,
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# TAB 2 — RATINGS
# ============================================================

with tab2:

    st.subheader(
        "⭐ IMDb Rating Analytics"
    )


    st.caption(
        "Explore IMDb rating patterns, categories and audience engagement."
    )


    st.markdown(
        "### 📊 IMDb Rating Distribution"
    )


    rating_distribution = (
        analytics_data[
            "averageRating"
        ]
        .dropna()
        .round(1)
        .value_counts()
        .sort_index()
    )


    rating_chart = pd.DataFrame(
        {
            "Rating":
                rating_distribution.index,

            "Movies":
                rating_distribution.values
        }
    )


    fig_rating = px.bar(
        rating_chart,
        x="Rating",
        y="Movies",
        title="Distribution of IMDb Ratings"
    )


    fig_rating.update_layout(
        template="plotly_dark",
        height=420
    )


    st.plotly_chart(
        fig_rating,
        use_container_width=True
    )


    # ========================================================
    # RATING CATEGORIES
    # ========================================================

    st.divider()

    st.subheader(
        "🏆 Rating Categories"
    )


    rating_categories = pd.cut(
        analytics_data[
            "averageRating"
        ],
        bins=[
            0,
            5,
            7,
            8,
            10
        ],
        labels=[
            "Low (<5)",
            "Average (5-6.9)",
            "Good (7-7.9)",
            "Excellent (8+)"
        ],
        include_lowest=True
    )


    category_counts = (
        rating_categories
        .value_counts()
        .reindex(
            [
                "Low (<5)",
                "Average (5-6.9)",
                "Good (7-7.9)",
                "Excellent (8+)"
            ],
            fill_value=0
        )
    )


    col1, col2, col3, col4 = st.columns(4)


    categories = [
        ("Low (<5)", "🔴"),
        ("Average (5-6.9)", "🟡"),
        ("Good (7-7.9)", "🟢"),
        ("Excellent (8+)", "🏆")
    ]


    columns = [
        col1,
        col2,
        col3,
        col4
    ]


    for column, (category, icon) in zip(
        columns,
        categories
    ):

        with column:

            st.metric(
                f"{icon} {category}",
                f"{category_counts[category]:,}"
            )


    # ========================================================
    # RATING SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "📌 Rating Summary"
    )


    rating_col1, rating_col2, rating_col3 = st.columns(3)


    with rating_col1:

        st.metric(
            "Average IMDb Rating",
            f"{analytics_data['averageRating'].mean():.2f}/10"
        )


    with rating_col2:

        st.metric(
            "Median IMDb Rating",
            f"{analytics_data['averageRating'].median():.1f}/10"
        )


    with rating_col3:

        high_rated_movies = analytics_data[
            analytics_data[
                "averageRating"
            ] >= 7.0
        ]


        high_rated_percentage = (
            len(high_rated_movies)
            /
            len(analytics_data)
            *
            100
        )


        st.metric(
            "Highly Rated Movies",
            f"{high_rated_percentage:.1f}%"
        )


    # ========================================================
    # TOP RATED
    # ========================================================

    st.divider()

    st.subheader(
        "🏅 Top Rated Movies"
    )


    st.caption(
        "Movies with at least 100 IMDb votes are used for more reliable ranking."
    )


    top_rated = (
        analytics_data[
            analytics_data[
                "numVotes"
            ] >= 100
        ]
        .sort_values(
            [
                "averageRating",
                "numVotes"
            ],
            ascending=[
                False,
                False
            ]
        )
        [
            [
                "primaryTitle",
                "startYear",
                "genres",
                "averageRating",
                "numVotes"
            ]
        ]
        .head(15)
        .copy()
    )


    top_rated.columns = [
        "Movie",
        "Year",
        "Genres",
        "IMDb Rating",
        "Votes"
    ]


    top_rated[
        "IMDb Rating"
    ] = top_rated[
        "IMDb Rating"
    ].round(1)


    st.dataframe(
        top_rated,
        hide_index=True,
        use_container_width=True
    )


    # ========================================================
    # MOST VOTED
    # ========================================================

    st.divider()

    st.subheader(
        "🗳️ Most Voted Movies"
    )


    most_voted = (
        analytics_data
        .sort_values(
            "numVotes",
            ascending=False
        )
        [
            [
                "primaryTitle",
                "startYear",
                "genres",
                "averageRating",
                "numVotes"
            ]
        ]
        .head(15)
        .copy()
    )


    most_voted.columns = [
        "Movie",
        "Year",
        "Genres",
        "IMDb Rating",
        "Votes"
    ]


    most_voted[
        "IMDb Rating"
    ] = most_voted[
        "IMDb Rating"
    ].round(1)


    st.dataframe(
        most_voted,
        hide_index=True,
        use_container_width=True
    )


    # ========================================================
    # INSIGHT
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 AI Data Insight"
    )


    st.info(
        f"""
        **What the data tells us**

        ⭐ Average IMDb rating:
        **{analytics_data['averageRating'].mean():.2f}/10**

        📊 Median IMDb rating:
        **{analytics_data['averageRating'].median():.1f}/10**

        🏆 Movies with rating ≥ 7.0:
        **{len(high_rated_movies):,}**

        🗳️ Vote count is important because movies with
        very few votes can have unstable ratings.

        🤖 Therefore, the recommendation engine combines
        similarity, rating, popularity and clustering.
        """
    )


# ============================================================
# TAB 3 — TIME ANALYSIS
# ============================================================

with tab3:

    st.subheader(
        "📅 Movie Production Timeline"
    )


    time_data = analytics_data.copy()


    time_data[
        "startYear"
    ] = pd.to_numeric(
        time_data[
            "startYear"
        ],
        errors="coerce"
    )


    time_data = time_data.dropna(
        subset=[
            "startYear"
        ]
    )


    time_data[
        "startYear"
    ] = time_data[
        "startYear"
    ].astype(int)


    time_data[
        "decade"
    ] = (
        time_data[
            "startYear"
        ]
        // 10
        * 10
    )


    earliest_year = (
        time_data[
            "startYear"
        ].min()
    )


    latest_year = (
        time_data[
            "startYear"
        ].max()
    )


    year_counts = (
        time_data[
            "startYear"
        ]
        .value_counts()
        .sort_index()
    )


    peak_year = (
        year_counts
        .idxmax()
    )


    peak_year_movies = (
        year_counts
        .max()
    )


    decade_counts = (
        time_data[
            "decade"
        ]
        .value_counts()
        .sort_index()
    )


    peak_decade = (
        decade_counts
        .idxmax()
    )


    peak_decade_movies = (
        decade_counts
        .max()
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "🎬 Earliest Movie",
            str(earliest_year)
        )


    with col2:

        st.metric(
            "🆕 Latest Movie",
            str(latest_year)
        )


    with col3:

        st.metric(
            "📈 Peak Year",
            str(peak_year),
            f"{peak_year_movies:,} movies"
        )


    with col4:

        st.metric(
            "🏆 Peak Decade",
            f"{peak_decade}s",
            f"{peak_decade_movies:,} movies"
        )


    # ========================================================
    # DECADE CHART
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Movies Released by Decade"
    )


    decade_chart = pd.DataFrame(
        {
            "Decade":
                decade_counts.index.astype(str),

            "Movies":
                decade_counts.values
        }
    )


    fig_decade = px.bar(
        decade_chart,
        x="Decade",
        y="Movies",
        title="Movie Production by Decade"
    )


    fig_decade.update_layout(
        template="plotly_dark",
        height=450
    )


    st.plotly_chart(
        fig_decade,
        use_container_width=True
    )


    # ========================================================
    # YEARLY CHART
    # ========================================================

    st.divider()

    st.subheader(
        "📈 Movies Released by Year"
    )


    yearly_chart = pd.DataFrame(
        {
            "Year":
                year_counts.index,

            "Movies":
                year_counts.values
        }
    )


    fig_year = px.line(
        yearly_chart,
        x="Year",
        y="Movies",
        title="Movie Production Over Time"
    )


    fig_year.update_layout(
        template="plotly_dark",
        height=450
    )


    st.plotly_chart(
        fig_year,
        use_container_width=True
    )


    # ========================================================
    # RECENT PRODUCTION
    # ========================================================

    st.divider()

    st.subheader(
        "🎥 Recent Movie Production"
    )


    recent_years = (
        year_counts
        .tail(15)
        .sort_index()
    )


    recent_chart = pd.DataFrame(
        {
            "Year":
                recent_years.index,

            "Movies":
                recent_years.values
        }
    )


    fig_recent = px.bar(
        recent_chart,
        x="Year",
        y="Movies",
        title="Recent 15-Year Production Trend"
    )


    fig_recent.update_layout(
        template="plotly_dark",
        height=420
    )


    st.plotly_chart(
        fig_recent,
        use_container_width=True
    )


    # ========================================================
    # DECADE SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "📋 Decade-wise Production Summary"
    )


    decade_summary = (
        time_data
        .groupby(
            "decade"
        )
        .agg(
            Movies=(
                "primaryTitle",
                "count"
            ),

            Average_Rating=(
                "averageRating",
                "mean"
            ),

            Average_Runtime=(
                "runtimeMinutes",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            "decade",
            ascending=False
        )
    )


    decade_summary.columns = [
        "Decade",
        "Movies",
        "Average IMDb Rating",
        "Average Runtime (min)"
    ]


    decade_summary[
        "Average IMDb Rating"
    ] = decade_summary[
        "Average IMDb Rating"
    ].round(2)


    decade_summary[
        "Average Runtime (min)"
    ] = decade_summary[
        "Average Runtime (min)"
    ].round(1)


    st.dataframe(
        decade_summary,
        hide_index=True,
        use_container_width=True
    )


    st.info(
        f"""
        **🧠 Time-Based AI Insight**

        🎬 Movies span from **{earliest_year}**
        to **{latest_year}**.

        📈 The peak production year is
        **{peak_year}**, with **{peak_year_movies:,} movies**.

        🏆 The peak production decade is
        **{peak_decade}s**, with
        **{peak_decade_movies:,} movies**.

        🤖 Production era information can help identify
        patterns in movie characteristics over time.
        """
    )


# ============================================================
# TAB 4 — POPULARITY
# ============================================================

with tab4:

    st.subheader(
        "🔥 Movie Popularity Analytics"
    )


    st.caption(
        "Analyze popularity using IMDb ratings and audience engagement."
    )


    popularity_data = analytics_data.copy()


    popularity_data[
        "popularity_score"
    ] = pd.to_numeric(
        popularity_data[
            "popularity_score"
        ],
        errors="coerce"
    )


    popularity_data[
        "averageRating"
    ] = pd.to_numeric(
        popularity_data[
            "averageRating"
        ],
        errors="coerce"
    )


    popularity_data[
        "numVotes"
    ] = pd.to_numeric(
        popularity_data[
            "numVotes"
        ],
        errors="coerce"
    )


    popularity_data = popularity_data.dropna(
        subset=[
            "popularity_score",
            "averageRating",
            "numVotes"
        ]
    )


    most_popular = popularity_data.loc[
        popularity_data[
            "popularity_score"
        ].idxmax()
    ]


    most_voted = popularity_data.loc[
        popularity_data[
            "numVotes"
        ].idxmax()
    ]


    highest_rating = popularity_data.loc[
        popularity_data[
            "averageRating"
        ].idxmax()
    ]


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🔥 Most Popular",
            most_popular[
                "primaryTitle"
            ]
        )

        st.caption(
            f"Popularity Score: "
            f"{most_popular['popularity_score']:,.0f}"
        )


    with col2:

        st.metric(
            "🗳️ Most Voted",
            most_voted[
                "primaryTitle"
            ]
        )

        st.caption(
            f"{int(most_voted['numVotes']):,} IMDb votes"
        )


    with col3:

        st.metric(
            "⭐ Highest Rated",
            highest_rating[
                "primaryTitle"
            ]
        )

        st.caption(
            f"IMDb Rating: "
            f"{highest_rating['averageRating']:.1f}/10"
        )


    # ========================================================
    # TOP POPULAR MOVIES
    # ========================================================

    st.divider()

    st.subheader(
        "🏆 Top Popular Movies"
    )


    popular_movies = (
        popularity_data
        .sort_values(
            "popularity_score",
            ascending=False
        )
        .head(15)
        .copy()
    )


    popular_display = popular_movies[
        [
            "primaryTitle",
            "startYear",
            "genres",
            "averageRating",
            "numVotes",
            "popularity_score"
        ]
    ].copy()


    popular_display.columns = [
        "Movie",
        "Year",
        "Genres",
        "IMDb Rating",
        "Votes",
        "Popularity Score"
    ]


    popular_display[
        "IMDb Rating"
    ] = popular_display[
        "IMDb Rating"
    ].round(1)


    popular_display[
        "Popularity Score"
    ] = popular_display[
        "Popularity Score"
    ].round(2)


    popular_display[
        "Votes"
    ] = popular_display[
        "Votes"
    ].map(
        lambda x: f"{int(x):,}"
    )


    st.dataframe(
        popular_display,
        hide_index=True,
        use_container_width=True
    )


    # ========================================================
    # RATING VS VOTES
    # ========================================================

    st.divider()

    st.subheader(
        "📊 IMDb Rating vs Audience Votes"
    )


    scatter_data = popularity_data.copy()


    if len(scatter_data) > 20000:

        scatter_data = scatter_data.sample(
            20000,
            random_state=42
        )


    fig_scatter = px.scatter(
        scatter_data,
        x="numVotes",
        y="averageRating",
        hover_name="primaryTitle",
        hover_data=[
            "startYear",
            "genres"
        ],
        title="IMDb Rating vs Number of Votes",
        labels={
            "numVotes":
                "Number of IMDb Votes",

            "averageRating":
                "IMDb Rating"
        },
        log_x=True,
        opacity=0.55
    )


    fig_scatter.update_layout(
        height=500,
        template="plotly_dark"
    )


    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )


    # ========================================================
    # POPULARITY DISTRIBUTION
    # ========================================================

    st.divider()

    st.subheader(
        "📈 Popularity Score Distribution"
    )


    popularity_chart_data = (
        popularity_data[
            "popularity_score"
        ]
        .clip(
            upper=
            popularity_data[
                "popularity_score"
            ].quantile(0.99)
        )
    )


    fig_popularity = px.histogram(
        popularity_chart_data,
        nbins=40,
        title="Distribution of Movie Popularity Scores",
        labels={
            "value":
                "Popularity Score",

            "count":
                "Number of Movies"
        }
    )


    fig_popularity.update_layout(
        height=420,
        template="plotly_dark"
    )


    st.plotly_chart(
        fig_popularity,
        use_container_width=True
    )


    # ========================================================
    # ENGAGEMENT
    # ========================================================

    st.divider()

    st.subheader(
        "🗳️ Movies With Highest Audience Engagement"
    )


    engagement = (
        popularity_data
        .sort_values(
            "numVotes",
            ascending=False
        )
        .head(10)
        [
            [
                "primaryTitle",
                "startYear",
                "averageRating",
                "numVotes"
            ]
        ]
        .copy()
    )


    engagement.columns = [
        "Movie",
        "Year",
        "IMDb Rating",
        "Votes"
    ]


    engagement[
        "IMDb Rating"
    ] = engagement[
        "IMDb Rating"
    ].round(1)


    engagement[
        "Votes"
    ] = engagement[
        "Votes"
    ].map(
        lambda x: f"{int(x):,}"
    )


    st.dataframe(
        engagement,
        hide_index=True,
        use_container_width=True
    )


    # ========================================================
    # POPULARITY INSIGHT
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 Popularity Insight"
    )


    popularity_correlation = (
        popularity_data[
            [
                "averageRating",
                "numVotes",
                "popularity_score"
            ]
        ]
        .corr()
        .loc[
            "averageRating",
            "numVotes"
        ]
    )


    st.info(
        f"""
        **What the data tells us**

        🔥 Popularity combines IMDb rating
        and audience engagement.

        ⭐ Rating alone does not completely represent
        a movie's popularity.

        🗳️ Vote count provides an important measure
        of audience engagement.

        📊 Rating-vote correlation:
        **{popularity_correlation:.3f}**

        🤖 The recommendation engine therefore considers
        similarity, rating, popularity and K-Means clustering.
        """
    )


# ============================================================
# AI MODEL PERFORMANCE DASHBOARD
# ============================================================

st.divider()

st.header(
    "🤖 AI Model Performance Dashboard"
)

st.markdown(
"""
Evaluate the Machine Learning algorithms used by the
movie recommendation system.
"""
)


# ============================================================
# CLASSIFICATION RESULTS
# ============================================================

classification_display = pd.DataFrame(
    {
        "Model": [
            "Decision Tree",
            "Random Forest",
            "Naive Bayes"
        ],

        "Accuracy": [
            0.7396,
            0.7728,
            0.2368
        ],

        "Precision": [
            0.4202,
            0.4624,
            0.5512
        ],

        "Recall": [
            0.7356,
            0.7146,
            0.2368
        ],

        "F1 Score": [
            0.5349,
            0.5615,
            0.1469
        ]
    }
)


# ============================================================
# CLASSIFICATION HEADER
# ============================================================

st.subheader(
    "📊 Classification Performance Comparison"
)


st.caption(
    "Comparison of Decision Tree, Random Forest and Naive Bayes."
)


# ============================================================
# BEST CLASSIFICATION MODEL
# ============================================================

best_index = (
    classification_display[
        "Accuracy"
    ].idxmax()
)


best_model = (
    classification_display.loc[
        best_index,
        "Model"
    ]
)


best_accuracy = (
    classification_display.loc[
        best_index,
        "Accuracy"
    ] * 100
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "🏆 Best Model",
        best_model
    )


with col2:

    st.metric(
        "🎯 Best Accuracy",
        f"{best_accuracy:.2f}%"
    )


with col3:

    st.metric(
        "🌲 Best F1 Score",
        f"{classification_display['F1 Score'].max():.4f}"
    )


# ============================================================
# CLASSIFICATION CHART
# ============================================================

fig_classification = px.bar(
    classification_display,
    x="Model",
    y=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    barmode="group",
    title="Classification Model Performance"
)


fig_classification.update_layout(
    template="plotly_dark",
    height=500,
    yaxis_title="Score",
    yaxis_range=[0, 1]
)


st.plotly_chart(
    fig_classification,
    use_container_width=True
)


# ============================================================
# CLASSIFICATION TABLE
# ============================================================

classification_table = (
    classification_display.copy()
)


for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]:

    classification_table[
        column
    ] = (
        classification_table[
            column
        ] * 100
    ).round(2).astype(str) + "%"


st.dataframe(
    classification_table,
    hide_index=True,
    use_container_width=True
)


st.success(
    f"""
    🏆 **Best Classification Model: {best_model}**

    The Random Forest model achieved the strongest overall
    classification accuracy at **77.28%**.
    """
)


# ============================================================
# REGRESSION PERFORMANCE
# ============================================================

st.divider()

st.subheader(
    "📈 Linear Regression Performance"
)


# Known model evaluation results

regression_mae = 0.7694
regression_rmse = 1.0156
regression_r2 = 0.3612


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "MAE",
        f"{regression_mae:.4f}"
    )


with col2:

    st.metric(
        "RMSE",
        f"{regression_rmse:.4f}"
    )


with col3:

    st.metric(
        "R² Score",
        f"{regression_r2:.4f}"
    )


regression_chart = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "RMSE",
            "R² Score"
        ],

        "Value": [
            regression_mae,
            regression_rmse,
            regression_r2
        ]
    }
)


fig_regression = px.bar(
    regression_chart,
    x="Metric",
    y="Value",
    title="Linear Regression Evaluation Metrics"
)


fig_regression.update_layout(
    template="plotly_dark",
    height=400
)


st.plotly_chart(
    fig_regression,
    use_container_width=True
)


st.info(
    """
    **📌 Interpretation**

    Linear Regression predicts the expected IMDb rating.

    Lower MAE and RMSE indicate smaller prediction errors,
    while a higher R² indicates that the model explains more
    variation in movie ratings.
    """
)


# ============================================================
# K-MEANS PERFORMANCE
# ============================================================

st.divider()

st.subheader(
    "🧩 K-Means Clustering Performance"
)


kmeans_clusters = 10
kmeans_silhouette = 0.1276


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "🧩 Number of Clusters",
        kmeans_clusters
    )


with col2:

    st.metric(
        "📐 Silhouette Score",
        f"{kmeans_silhouette:.4f}"
    )


kmeans_chart = pd.DataFrame(
    {
        "Metric": [
            "Silhouette Score"
        ],

        "Value": [
            kmeans_silhouette
        ]
    }
)


fig_kmeans = px.bar(
    kmeans_chart,
    x="Metric",
    y="Value",
    title="K-Means Clustering Quality"
)


fig_kmeans.update_layout(
    template="plotly_dark",
    height=380,
    yaxis_range=[0, 1]
)


st.plotly_chart(
    fig_kmeans,
    use_container_width=True
)


st.info(
    """
    **📌 Interpretation**

    The Silhouette Score measures how well movies fit within
    their assigned clusters compared with other clusters.

    A higher score generally indicates better separation
    between clusters.
    """
)


# ============================================================
# COMPLETE AI MODEL SUMMARY
# ============================================================

st.divider()

st.subheader(
    "🧠 Complete AI Algorithm Summary"
)


model_summary = pd.DataFrame(
    {
        "Algorithm": [
            "KNN",
            "K-Means",
            "Decision Tree",
            "Random Forest",
            "Linear Regression",
            "Naive Bayes"
        ],

        "AI Task": [
            "Movie Similarity",
            "Movie Clustering",
            "Rating Classification",
            "Advanced Classification",
            "Rating Prediction",
            "Rating Category Classification"
        ],

        "Role in System": [
            "Finds similar movies",
            "Groups movies by characteristics",
            "Predicts high-rated category",
            "Predicts high-rated category + probability",
            "Predicts expected IMDb rating",
            "Predicts rating category"
        ]
    }
)


st.dataframe(
    model_summary,
    hide_index=True,
    use_container_width=True
)


# ============================================================
# FINAL AI SYSTEM INSIGHT
# ============================================================

st.divider()

st.subheader(
    "🚀 Overall AI System Insight"
)


st.markdown(
"""
<div class="section-card">

<b>🎯 Why this system is a Hybrid AI system</b>

<br><br>

The system does not depend on a single Machine Learning
algorithm.

<br><br>

<b>🔍 KNN</b>
finds movies with similar characteristics.

<br><br>

<b>🧩 K-Means</b>
groups movies into clusters based on their features.

<br><br>

<b>🌳 Decision Tree</b>
classifies movies into rating categories.

<br><br>

<b>🌲 Random Forest</b>
provides stronger classification and high-rating probability.

<br><br>

<b>📈 Linear Regression</b>
predicts an expected IMDb rating.

<br><br>

<b>🧠 Naive Bayes</b>
classifies movies into rating categories.

<br><br>

<b>🤖 Hybrid Recommendation Engine</b>
combines KNN similarity, rating, popularity and cluster
information to produce the final recommendation ranking.

</div>
""",
unsafe_allow_html=True
)


# ============================================================
# ABOUT PROJECT
# ============================================================

st.divider()

st.subheader(
    "📌 About the Project"
)


st.write(
"""
This AI-based movie recommendation system uses real movie
data from the official IMDb dataset and applies multiple
Machine Learning algorithms.

The system uses KNN for movie similarity, K-Means for movie
clustering, Decision Tree and Random Forest for classification,
Linear Regression for rating prediction, and Naive Bayes for
rating-category classification.

The Hybrid AI recommendation engine combines movie similarity,
IMDb ratings, popularity and cluster information to produce
intelligent movie recommendations.

The dashboard provides interactive analysis of movie genres,
ratings, production trends, popularity and Machine Learning
model performance.

Movie posters, movie details, trailers and watch-provider availability are fetched
from TMDB only for dashboard presentation and do not modify the IMDb ML dataset.

Watch-provider data is supplied by JustWatch through TMDB and is intended to
show legal streaming, rental and purchase availability.

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
"""
<div style="
text-align:center;
padding:20px;
opacity:0.55;
">

🎬 <b>AI Movie Intelligence</b>

<br>

AI-Based Movie Recommendation System

<br>

Machine Learning • Artificial Intelligence • IMDb Dataset

<br>

Movie artwork provided by TMDB. This product uses the TMDB API
but is not endorsed or certified by TMDB.

</div>
""",
unsafe_allow_html=True
)