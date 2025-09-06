import re
import pandas as pd
from transformers import pipeline

# -----------------------
# Load Sentiment Model
# -----------------------
sentiment_model = pipeline("sentiment-analysis")

# -----------------------
# Mood Detection Function
# -----------------------
def predict_moods_from_text(text, model):
    text_lower = text.lower()
    detected_moods = set()
    excluded_genres = set()

    # Mood keyword map
    mood_map = {
        "happy": ["funny", "happy", "comedy", "lighthearted", "feel good", "cheerful", "upbeat", "feel better"],
        "romantic": ["love", "romantic", "romance", "heartwarming", "date night"],
        "excited": ["action", "thrill", "adventure", "thriller", "exciting", "fast-paced", "epic"],
        "scared": ["horror", "scared", "spooky", "mystery", "suspenseful", "creepy", "ghost", "zombie"],
        "thoughtful": ["documentary", "history", "intelligent", "thoughtful", "mind-bending", "true story"],
        "sad": ["sad", "drama", "cry", "emotional", "tragic", "heartbreaking", "tearjerker"]
    }

    # Keyword → Genre map for exclusions
    keyword_to_genre = {
        "comedy": "Comedy", "romance": "Romance", "action": "Action", "thriller": "Thriller",
        "horror": "Horror", "mystery": "Mystery", "documentary": "Documentary", "history": "History",
        "drama": "Drama"
    }

    # Negation patterns
    negations = [r'\bnot\b', r'anything but', r"don't want", r'without']

    # --- Detect exclusions ---
    for keyword, genre in keyword_to_genre.items():
        if any(re.search(f"{pattern}.*\\b{keyword}\\b", text_lower) for pattern in negations):
            excluded_genres.add(genre)

    # --- Detect moods (multi-label) ---
    for mood, keywords in mood_map.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_moods.add(mood)

    # --- Fallback if nothing detected ---
    if not detected_moods:
        sentiment = model(text)[0]['label']
        final_moods = ['happy'] if sentiment == 'POSITIVE' else ['sad']
    else:
        final_moods = list(detected_moods)

    return final_moods, list(excluded_genres)

# -----------------------
# Movie Recommendation Function
# -----------------------
def recommend_movies(user_input, df, model):
    moods, excluded = predict_moods_from_text(user_input, model)

    print(f"\nDetected moods: {moods}")
    print(f"Excluded genres: {excluded}")

    recommendations = []

    for mood in moods:
        filtered = df.copy()

        # Exclude unwanted genres
        if excluded:
            for genre in excluded:
                filtered = filtered[~filtered['genres'].str.contains(genre, case=False, na=False)]

        # Pick top 3 from this mood
        mood_movies = filtered[filtered['mood'] == mood].head(3).to_dict(orient="records")
        recommendations.extend(mood_movies)

    return recommendations

# -----------------------
# Example Movies Dataset
# -----------------------
data = [
    {"title": "The Hangover", "genres": "Comedy", "mood": "happy"},
    {"title": "La La Land", "genres": "Romance, Musical", "mood": "romantic"},
    {"title": "Mad Max: Fury Road", "genres": "Action, Adventure", "mood": "excited"},
    {"title": "The Conjuring", "genres": "Horror, Mystery", "mood": "scared"},
    {"title": "Inside Out", "genres": "Animation, Drama", "mood": "thoughtful"},
    {"title": "The Fault in Our Stars", "genres": "Romance, Drama", "mood": "sad"},
]
df = pd.DataFrame(data)

# -----------------------
# Run Test
# -----------------------
if __name__ == "__main__":
    query = input("What kind of movie do you want to watch? ")
    recs = recommend_movies(query, df, sentiment_model)

    print("\n🎬 Recommended Movies:")
    for movie in recs:
        print(f"- {movie['title']} ({movie['genres']})")
