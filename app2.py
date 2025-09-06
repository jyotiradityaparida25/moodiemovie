import streamlit as st
import pandas as pd
import ast
import re
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('moodie_movie_data.csv.xz')
        for col in ['genres', 'cast', 'director']:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        df['overview'] = df['overview'].fillna('')
        df['soup'] = df.apply(lambda x: ' '.join(x['genres']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['director']) + ' ' + x['overview'], axis=1)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['soup']).astype('float32')
        indices = pd.Series(df.index, index=df['title'])
        return df, tfidf_matrix, indices
    except FileNotFoundError:
        st.error("Error: `moodie_movie_data.csv.xz` not found. Please ensure it's in your project folder.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None, None, None

@st.cache_resource
def get_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_user_intent(query, movie_titles):
    query_lower = query.lower()
    match, score = process.extractOne(query, movie_titles)
    if score > 85:
        return {'type': 'title', 'entity': match}
    person_keywords = ['starring', 'with', 'actor', 'actress', 'directed by', 'director']
    if any(keyword in query_lower for keyword in person_keywords):
        entity = re.sub('|'.join(person_keywords), '', query_lower, flags=re.IGNORECASE).strip()
        return {'type': 'person', 'entity': entity.title()}
    return {'type': 'mood', 'entity': query}

def predict_moods_from_text(text, model):
    text_lower = text.lower()
    detected = set()
    
    # --- FIX: Expanded the keyword list to include genre names ---
    mood_map = {
        "happy": ["funny", "happy", "comedy", "lighthearted", "feel better"],
        "romantic": ["love", "romantic", "romance"],
        "excited": ["action", "thrill", "adventure", "thriller"],
        "scared": ["horror", "scared", "spooky", "mystery"],
        "thoughtful": ["documentary", "history", "intelligent", "thoughtful"],
        "sad": ["sad", "drama", "cry", "emotional"]
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
    
    # Exclude genres from opposite moods if possible
    if "happy" in moods and "scared" not in moods:
        recs = recs[~recs['genres'].apply(lambda x: 'Horror' in x)]
    if "scared" in moods and "happy" not in moods:
        recs = recs[~recs['genres'].apply(lambda x: 'Comedy' in x)]
        
    return recs.nlargest(3, 'vote_count')

def get_recommendations_by_person(name, df):
    name_lower = name.lower()
    is_actor = df['cast'].apply(lambda x: name_lower in [a.lower() for a in x])
    is_director = df['director'].apply(lambda x: name_lower in [d.lower() for d in x])
    recs = df[is_actor | is_director]
    return recs.nlargest(5, 'vote_count')

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

def display_recs(recs, message="", context_key=""):
    if recs.empty:
        st.warning("Sorry, couldn't find any movies for that.")
        return

    st.subheader(message)
    for i, (_, movie) in enumerate(recs.iterrows()):
        with st.expander(f"**{movie['title']}**"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**Rating:** ⭐ {movie['vote_average']:.1f}/10 ({int(movie['vote_count'])} votes)")
                st.markdown(f"**Genres:** {', '.join(movie['genres'])}")
                if movie['director']: st.markdown(f"**Director:** {', '.join(movie['director'])}")
                if movie['cast']: st.markdown(f"**Cast:** {', '.join(movie['cast'])}")
            with col2:
                st.markdown("**Overview:**")
                st.write(movie['overview'])
                if st.button("More like this", key=f"{context_key}_{i}_{movie['id']}"):
                    st.session_state.run_similar = movie['title']
                    st.rerun()

setup_page()
st.title("🤖 MoodieMovie")

df, tfidf_matrix, indices = load_data()
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
        st.session_state.run_similar = None
        st.rerun()
        
    with st.expander("📖 How to Use"):
        st.markdown("""
        **1. Search by Mood**
        - *"I want a funny movie"*
        - *"a thriller but not horror"*
        **2. Search by Title/Person**
        - *"The Dark Knight"*
        - *"movies starring Tom Hanks"*
        **3. Discover Similar Movies**
        Click the **"More like this"** button on any movie.
        """)

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "recommendations" in msg:
            display_recs(pd.DataFrame(msg["recommendations"]), context_key=f"msg_{i}")

if st.session_state.run_similar:
    title_to_match = st.session_state.run_similar
    st.session_state.run_similar = None
    
    st.session_state.messages.append({"role": "user", "content": f"Show me movies similar to '{title_to_match}'."})
    st.rerun()

user_input = st.chat_input("What would you like to watch?")

if user_input and "last_input" not in st.session_state or st.session_state.last_input != user_input:
    st.session_state.last_input = user_input
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            intent = get_user_intent(user_input, indices.index)
            recs = pd.DataFrame()
            message = ""

            if intent['type'] == 'title':
                recs = df[df['title'] == intent['entity']]
                message = f"Here are the details for **{intent['entity']}**:"
            elif intent['type'] == 'person':
                recs = get_recommendations_by_person(intent['entity'], df)
                message = f"Top movies featuring **{intent['entity']}**:"
            elif intent['type'] == 'mood':
                moods = predict_moods_from_text(intent['entity'], sentiment_model)
                recs = get_recommendations_by_mood(moods, df)
                message = f"Here are some **{' and '.join(moods)}** movies for you:"

            display_recs(recs, message, context_key="new_rec")
            
            bot_msg = {"role": "assistant", "content": message}
            if not recs.empty:
                bot_msg["recommendations"] = recs.to_dict('records')
            st.session_state.messages.append(bot_msg)
    st.rerun()
