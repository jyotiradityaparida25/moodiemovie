import streamlit as st
import pandas as pd
import time
import ast
import re
import os
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

@st.cache_data
def load_and_prepare_data():
    """
    Loads and prepares the TMDb metadata, creating a TF-IDF matrix from genres and overview.
    """
    try:
        # Load only the metadata dataset
        meta_df = pd.read_csv('movies_metadata.csv.gz', compression='gzip', low_memory=False)

        # Clean and filter the data
        meta_df = meta_df[meta_df['id'].str.isnumeric()]
        meta_df['id'] = meta_df['id'].astype(int)
        
        meta_df['vote_count'] = pd.to_numeric(meta_df['vote_count'], errors='coerce').fillna(0)
        df = meta_df[meta_df['vote_count'] >= 100].copy() 

        # Feature Engineering for Genres
        def safe_literal_eval(s):
            try: return ast.literal_eval(s)
            except: return []
        df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in safe_literal_eval(x)])
        
        # Select final columns (no cast or director)
        df = df[['id', 'title', 'overview', 'genres', 'poster_path', 'vote_average', 'vote_count']]
        df['overview'] = df['overview'].fillna('')
        df.dropna(subset=['title', 'poster_path'], inplace=True)

        # Create a "soup" from only genres and overview
        df['soup'] = df.apply(lambda x: ' '.join(x['genres']) + ' ' + x['overview'], axis=1)

        # NLP Model Training (TF-IDF)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['soup']).astype('float32')
        
        indices = pd.Series(df.index, index=df['title'])

        return df, tfidf_matrix, indices

    except FileNotFoundError as e:
        st.error(f"Error: A required data file was not found. Please check for `{e.filename}` in your repo.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None, None, None

@st.cache_resource
def get_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_user_intent(query, movie_titles):
    """Classifies user intent as either title or mood search."""
    query_lower = query.lower()
    
    # Check for a direct title match
    match, score = process.extractOne(query, movie_titles)
    if score > 85:
        return {'type': 'title', 'entity': match}

    # If not a title, it's a mood search
    return {'type': 'mood', 'entity': query}

def predict_moods_from_text(text, model):
    text_lower = text.lower()
    detected = set()
    mood_map = {
        "happy": ["funny", "happy", "feel better", "lighthearted"],
        "romantic": ["love", "romantic"],
        "excited": ["action", "thrill", "adventure"],
        "scared": ["horror", "scared", "spooky"],
        "thoughtful": ["documentary", "history", "intelligent"],
        "sad": ["sad", "drama", "cry"]
    }
    negations = [r'\bnot\b', r'anything but', r'don\'t want', r'without']
    
    excluded = set()
    for mood, keywords in mood_map.items():
        for keyword in keywords:
            if any(re.search(f"{pattern}.*\\b{keyword}\\b", text_lower) for pattern in negations):
                excluded.add(mood)

    for mood, keywords in mood_map.items():
        if any(keyword in text_lower for keyword in keywords):
            detected.add(mood)

    final_moods = list(detected - excluded)
    
    if not final_moods:
        return ['happy'] if model(text)[0]['label'] == 'POSITIVE' else ['sad']
    return final_moods

def get_recommendations_by_mood(moods, df):
    genre_map = {
        "happy": ["Comedy", "Romance"], "excited": ["Action", "Thriller"], "sad": ["Drama"],
        "scared": ["Horror", "Mystery"], "romantic": ["Romance"], "thoughtful": ["Documentary", "History"]
    }
    target_genres = set(g for m in moods for g in genre_map.get(m, []))
    if not target_genres: 
        return pd.DataFrame()
    recs = df[df['genres'].apply(lambda x: not target_genres.isdisjoint(x))]
    return recs.nlargest(3, 'vote_count')

def get_similar_content(title, df, matrix, indices):
    if title not in indices: 
        return pd.DataFrame()
    
    lookup = indices[title]
    idx = lookup.iloc[0] if isinstance(lookup, pd.Series) else lookup

    if idx >= matrix.shape[0]: return pd.DataFrame()
        
    sim_scores = cosine_similarity(matrix[idx], matrix)
    sim_scores = sorted(list(enumerate(sim_scores[0])), key=lambda x: x[1], reverse=True)[1:6]
    
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

def setup_page():
    st.set_page_config(page_title="MoodieMovie", page_icon="🤖", layout="wide")
    st.markdown("""
        <style>
            .stApp { background-color: #1a1a2e; color: #e0e0e0; }
            h1 { color: #e94560; text-align: center; }
            [data-testid="stChatMessage"] { background-color: #16213e; border-radius: 10px; }
            [data-testid="stExpander"] { border-radius: 10px; border: 1px solid #0f3460; }
        </style>
    """, unsafe_allow_html=True)

def display_recs(recs, message=""):
    if recs.empty:
        st.warning("Sorry, couldn't find any movies for that.")
        return

    st.subheader(message)
    for _, movie in recs.iterrows():
        with st.expander(f"**{movie['title']}**"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**Rating:** ⭐ {movie['vote_average']:.1f}/10 ({int(movie['vote_count'])} votes)")
                st.markdown(f"**Genres:** {', '.join(movie['genres'])}")
            with col2:
                st.markdown("**Overview:**")
                st.write(movie['overview'])
                if st.button("More like this", key=f"more_{movie['id']}"):
                    st.session_state.run_similar = movie['title']
                    st.rerun()

setup_page()
st.title("🤖 MoodieMovie")

df, tfidf_matrix, indices = load_and_prepare_data()
if df is None:
    st.stop()
sentiment_model = get_model()

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How are you feeling, or what are you looking for?"}]
if 'run_similar' not in st.session_state:
    st.session_state.run_similar = None

with st.sidebar:
    st.title("About MoodieMovie")
    if df is not None:
        st.write(f"Loaded **{df['title'].nunique()}** unique movies.")
    if st.button("Clear Conversation"):
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How are you feeling, or what are you looking for?"}]
        st.rerun()
        
    with st.expander("📖 How to Use"):
        st.markdown("""
        **1. Search by Mood**
        - *"I want a funny movie"*
        - *"a thriller but not horror"*
        **2. Search by Title**
        - *"The Dark Knight"*
        **3. Discover Similar Movies**
        Click the **"More like this"** button on any movie.
        """)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "recommendations" in msg:
            display_recs(pd.DataFrame(msg["recommendations"]))

if st.session_state.run_similar:
    similar_to_title = st.session_state.run_similar
    st.session_state.run_similar = None
    
    st.session_state.messages.append({"role": "user", "content": f"Show me movies like '{similar_to_title}'."})
    
    with st.chat_message("assistant"):
        with st.spinner(f"Finding movies like '{similar_to_title}'..."):
            recs = get_similar_content(similar_to_title, df, tfidf_matrix, indices)
            message = f"If you liked {similar_to_title}, you might also like:"
            display_recs(recs, message)
            
            bot_msg = {"role": "assistant", "content": message, "recommendations": recs.to_dict('records')}
            st.session_state.messages.append(bot_msg)
    st.rerun()

user_input = st.chat_input("What would you like to watch?")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            intent = get_user_intent(user_input, indices.index)
            recs = pd.DataFrame()
            message = ""

            if intent['type'] == 'title':
                recs = df[df['title'] == intent['entity']]
                message = f"Here are the details for **{intent['entity']}**:"
            elif intent['type'] == 'mood':
                moods = predict_moods_from_text(intent['entity'], sentiment_model)
                recs = get_recommendations_by_mood(moods, df)
                message = f"Here are some **{' and '.join(moods)}** movies for you:"

            display_recs(recs, message)
            
            bot_msg = {"role": "assistant", "content": message}
            if not recs.empty:
                bot_msg["recommendations"] = recs.to_dict('records')
            st.session_state.messages.append(bot_msg)
