import streamlit as st
import pandas as pd
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process

# ------------------------
# Load Data
# ------------------------
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
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# ------------------------
# Keyword-based Mood Detector
# ------------------------
def predict_moods_from_text(text):
    text_lower = text.lower()
    detected_moods = set()
    excluded_genres = set()

    mood_map = {
        "happy": ["funny", "happy", "comedy", "lighthearted", "cheerful"],
        "romantic": ["love", "romantic", "romance", "heartwarming"],
        "excited": ["action", "thrill", "adventure", "thriller", "epic"],
        "scared": ["horror", "scary", "spooky", "mystery", "suspense"],
        "thoughtful": ["documentary", "history", "intelligent", "mind-bending"],
        "sad": ["sad", "drama", "cry", "tearjerker", "tragic"]
    }

    keyword_to_genre = {
        "comedy": "Comedy", "romance": "Romance", "action": "Action", "thriller": "Thriller",
        "horror": "Horror", "mystery": "Mystery", "documentary": "Documentary", "history": "History",
        "drama": "Drama"
    }

    negations = [r'\bnot\b', r'without', r"don't want", r'no ']

    # Exclusion detection
    for keyword, genre in keyword_to_genre.items():
        if any(re.search(f"{pattern}.*\\b{keyword}\\b", text_lower) for pattern in negations):
            excluded_genres.add(genre)

    # Mood detection
    for mood, keywords in mood_map.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_moods.add(mood)

    if not detected_moods:
        detected_moods = {"happy"}  # default mood if unclear

    return list(detected_moods), list(excluded_genres)

# ------------------------
# Recommendation Functions
# ------------------------
def get_recommendations_by_mood(moods, df, excluded_genres=None):
    genre_map = {
        "happy": ["Comedy", "Romance"], "excited": ["Action", "Thriller"], "sad": ["Drama"],
        "scared": ["Horror", "Mystery"], "romantic": ["Romance"], "thoughtful": ["Documentary", "History"]
    }

    required_genres = set(g for m in moods for g in genre_map.get(m, []))
    forbidden_genres = set(excluded_genres) if excluded_genres else set()
    required_genres -= forbidden_genres

    if not required_genres:
        return pd.DataFrame()

    def is_valid_movie(movie_genres):
        gset = set(movie_genres)
        return (not gset.isdisjoint(required_genres)) and gset.isdisjoint(forbidden_genres)

    recs = df[df['genres'].apply(is_valid_movie)]
    return recs.nlargest(5, 'vote_count')

def get_recommendations_by_person(name, df):
    name_lower = name.lower()
    is_actor = df['cast'].apply(lambda x: name_lower in [a.lower() for a in x])
    is_director = df['director'].apply(lambda x: name_lower in [d.lower() for d in x])
    return df[is_actor | is_director].nlargest(5, 'vote_count')

def get_similar_content(title, df, matrix, indices):
    if title not in indices: 
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = cosine_similarity(matrix[idx], matrix)
    sim_scores = sorted(list(enumerate(sim_scores[0])), key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

# ------------------------
# UI
# ------------------------
def main():
    st.set_page_config(page_title="MoodieMovie", page_icon="🎬", layout="wide")
    st.title("🎬 MoodieMovie – Fast Movie Recommender")

    df, tfidf_matrix, indices = load_data()
    if df is None:
        st.stop()

    # Chat-style UI
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How are you feeling, or what are you looking for?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "recommendations" in msg:
                recs = pd.DataFrame(msg["recommendations"])
                for _, movie in recs.iterrows():
                    st.markdown(f"**{movie['title']}** ({', '.join(movie['genres'])}) ⭐ {movie['vote_average']:.1f}")

    if user_input := st.chat_input("Tell me your mood, a movie, or an actor..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            intent = None
            match, score = process.extractOne(user_input, indices.index)
            if score > 85:
                intent = {"type": "title", "entity": match}
            elif any(k in user_input.lower() for k in ["actor", "starring", "directed by", "with"]):
                name = re.sub(r"(actor|starring|directed by|with)", "", user_input, flags=re.IGNORECASE).strip()
                intent = {"type": "person", "entity": name}
            else:
                intent = {"type": "mood", "entity": user_input}

            recs, message = pd.DataFrame(), ""
            if intent['type'] == 'title':
                recs = df[df['title'] == intent['entity']]
                message = f"Here are movies similar to **{intent['entity']}**:"
                recs = get_similar_content(intent['entity'], df, tfidf_matrix, indices)
            elif intent['type'] == 'person':
                recs = get_recommendations_by_person(intent['entity'], df)
                message = f"Top movies featuring **{intent['entity']}**:"
            elif intent['type'] == 'mood':
                moods, excluded = predict_moods_from_text(intent['entity'])
                recs = get_recommendations_by_mood(moods, df, excluded)
                message = f"Here are some **{' and '.join(moods)}** movies (excluding {', '.join(excluded)})"

            if recs.empty:
                st.warning("No perfect matches found.")
            else:
                for _, movie in recs.iterrows():
                    st.markdown(f"**{movie['title']}** ({', '.join(movie['genres'])}) ⭐ {movie['vote_average']:.1f}")

            st.session_state.messages.append({"role": "assistant", "content": message, "recommendations": recs.to_dict("records")})

if __name__ == "__main__":
    main()
