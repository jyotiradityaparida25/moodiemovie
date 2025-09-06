import streamlit as st
import pandas as pd
import time
import ast
import re
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv('moodie_movie_data.csv.xz')
        
        for col in ['genres', 'cast', 'director']:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        
        df['overview'] = df['overview'].fillna('')
        df.dropna(subset=['title'], inplace=True)
        
        df['soup'] = df.apply(lambda x: ' '.join(x['genres']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['director']) + ' ' + x['overview'], axis=1)
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['soup']).astype('float32')
        
        indices = pd.Series(df.index, index=df['title'])

        return df, tfidf_matrix, indices

    except FileNotFoundError as e:
        st.error(f"Error: A required data file was not found. Please check for `moodie_movie_data.csv.xz` in the correct directory.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None, None, None

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_intent(query, movie_titles):
    query_lower = query.lower()
    match, score = process.extractOne(query, movie_titles)
    if score > 85:
        return {'type': 'title_search', 'entity': match}
    person_keywords = ['starring', 'with', 'actor', 'actress', 'directed by', 'director']
    if any(keyword in query_lower for keyword in person_keywords):
        entity = re.sub('|'.join(person_keywords), '', query_lower, flags=re.IGNORECASE).strip()
        return {'type': 'person_search', 'entity': entity.title()}
    return {'type': 'mood_search', 'entity': query}

def predict_moods(text, sentiment_pipeline):
    text_lower = text.lower()
    
    mood_to_genre = {
        "happy": {"keywords": ["funny", "happy", "comedy", "lighthearted", "upbeat"], "genres": ["Comedy"]},
        "romantic": {"keywords": ["love", "romantic", "romance", "date night"], "genres": ["Romance"]},
        "excited": {"keywords": ["action", "thrill", "adventure", "exciting", "fast-paced"], "genres": ["Action", "Adventure", "Thriller"]},
        "scared": {"keywords": ["horror", "scary", "spooky", "creepy", "suspenseful"], "genres": ["Horror", "Mystery", "Thriller"]},
        "thoughtful": {"keywords": ["documentary", "history", "biography", "thoughtful", "intellectual"], "genres": ["Documentary", "History"]},
        "sad": {"keywords": ["sad", "emotional", "drama", "cry", "melancholy"], "genres": ["Drama"]}
    }

    detected_moods = set()
    excluded_genres = set()
    
    negation_patterns = [r'\bnot\b', r'\bbut not\b', r'anything but', r'don\'t want', r'without']

    for mood, data in mood_to_genre.items():
        for keyword in data["keywords"]:
            if any(re.search(f"{pattern}.*\\b{keyword}\\b", text_lower) for pattern in negation_patterns):
                for genre in data["genres"]:
                    excluded_genres.add(genre)

    for mood, data in mood_to_genre.items():
        if any(keyword in text_lower for keyword in data["keywords"]):
            detected_moods.add(mood)

    final_moods = [m for m in detected_moods if m not in [k for k,v in mood_to_genre.items() if any(g in excluded_genres for g in v['genres'])]]

    if not final_moods:
        sentiment = sentiment_pipeline(text)[0]['label']
        if sentiment == 'POSITIVE':
            final_moods = ['happy']
        elif sentiment == 'NEGATIVE':
            final_moods = ['sad']

    return final_moods, list(excluded_genres)

def get_mood_recommendations(moods, df, excluded_genres):
    mood_to_genre = {
        "happy": ["Comedy", "Romance"], "excited": ["Action", "Adventure", "Thriller"], "sad": ["Drama"],
        "scared": ["Horror", "Mystery"], "romantic": ["Romance"], "thoughtful": ["Documentary", "History"]
    }
    required_genres = set(g for m in moods for g in mood_to_genre.get(m, []))
    forbidden_genres = set(excluded_genres)
    
    if not required_genres:
        return pd.DataFrame()
    
    recs = df[
        df['genres'].apply(lambda x: not set(x).isdisjoint(required_genres)) &
        df['genres'].apply(lambda x: set(x).isdisjoint(forbidden_genres))
    ]
    return recs.nlargest(3, 'vote_count')

def get_person_recommendations(person_name, df):
    person_name_lower = person_name.lower()
    is_actor = df['cast'].apply(lambda x: person_name_lower in [actor.lower() for actor in x])
    is_director = df['director'].apply(lambda x: person_name_lower in [director.lower() for director in x])
    recs = df[is_actor | is_director]
    return recs.nlargest(5, 'vote_count')

def get_content_recommendations(title, df, tfidf_matrix, indices):
    if title not in indices.index: 
        return pd.DataFrame()
    lookup_result = indices[title]
    idx = lookup_result.iloc[0] if isinstance(lookup_result, pd.Series) else lookup_result
    if idx >= tfidf_matrix.shape[0]: return pd.DataFrame()
    movie_vector = tfidf_matrix[idx]
    sim_scores = cosine_similarity(movie_vector, tfidf_matrix)
    sim_scores = list(enumerate(sim_scores[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

def setup_page_config_and_styles():
    st.set_page_config(page_title="MoodieMovie", page_icon="🤖", layout="wide")
    st.markdown("""
        <style>
            .stApp { background-color: #1a1a2e; color: #e0e0e0; }
            h1 { color: #e94560; text-align: center; }
            [data-testid="stChatMessage"] { background-color: #16213e; border-radius: 10px; }
            [data-testid="stExpander"] { border-radius: 10px; border: 1px solid #0f3460; }
        </style>
    """, unsafe_allow_html=True)

def display_movies(movies_df, message="", context_key=""):
    if movies_df.empty:
        st.warning("Sorry, I couldn't find any movies that fit that request.")
        return
    st.subheader(message)
    for _, movie in movies_df.iterrows():
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
                if st.button("More like this", key=f"more_{context_key}_{movie['id']}"):
                    st.session_state.run_more_like_this = movie['title']
                    st.rerun()

def main():
    setup_page_config_and_styles()
    st.title("🤖 MoodieMovie")
    df, tfidf_matrix, indices = load_and_prepare_data()
    if df is None: return
    sentiment_pipeline = load_sentiment_model()
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me for a movie by title, mood, or person."}]
    if 'run_more_like_this' not in st.session_state:
        st.session_state.run_more_like_this = None
    with st.sidebar:
        st.title("About CineBot AI")
        if not df.empty:
            st.write(f"Loaded **{df['title'].nunique()}** unique movies.")
        if st.button("Clear Conversation"):
            st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me for a movie by title, mood, or person."}]
            st.rerun()
        with st.expander("📖 How to Use CineBot AI"):
            st.markdown("""
            This chatbot can understand three different types of requests:

            **1. Search by Mood**
            Describe a feeling or genre. You can even use negations!
            - *e.g., "I want to watch a funny movie"*
            - *e.g., "a thriller but not horror"*
            
            **2. Discover Similar Movies**
            After getting a recommendation, click the **"More like this"** button to find movies with a similar theme and style.
            """)
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "recommendations" in msg:
                display_movies(pd.DataFrame(msg["recommendations"]), context_key=f"msg_{i}")
    if st.session_state.run_more_like_this:
        title_to_match = st.session_state.run_more_like_this
        st.session_state.run_more_like_this = None
        user_msg = f"Show me movies similar to '{title_to_match}'."
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("assistant"):
            with st.spinner(f"Finding movies like '{title_to_match}'..."):
                recs = get_content_recommendations(title_to_match, df, tfidf_matrix, indices)
                message = f"If you liked {title_to_match}, you might also like:"
                display_movies(recs, message, context_key="similar")
                bot_msg = {"role": "assistant", "content": message, "recommendations": recs.to_dict('records')}
                st.session_state.messages.append(bot_msg)
        st.rerun()
    user_input = st.chat_input("What would you like to watch?")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                intent = get_intent(user_input, indices.index)
                recs_df, message = pd.DataFrame(), ""
                if intent['type'] == 'title_search':
                    recs_df = df[df['title'] == intent['entity']]
                    message = f"Here are the details for **{intent['entity']}**:"
                elif intent['type'] == 'person_search':
                    recs_df = get_person_recommendations(intent['entity'], df)
                    message = f"Top-rated movies featuring **{intent['entity']}**:"
                elif intent['type'] == 'mood_search':
                    moods, excluded_genres = predict_moods(intent['entity'], sentiment_pipeline)
                    recs_df = get_mood_recommendations(moods, df, excluded_genres)
                    message = f"Here are some **{' and '.join(moods)}** movies for you:"
                display_movies(recs_df, message, context_key="new_search")
                bot_msg = {"role": "assistant", "content": message}
                if not recs_df.empty:
                    bot_msg["recommendations"] = recs_df.to_dict('records')
                st.session_state.messages.append(bot_msg)
                st.rerun()

if __name__ == "__main__":
    main()
