import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# STEP 1: LOAD DATA
# =========================

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Select important columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# Remove missing values
movies.dropna(inplace=True)

# =========================
# STEP 2: CONVERT STRING TO LIST
# =========================

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# =========================
# STEP 3: CONVERT OVERVIEW TO LIST
# =========================

movies['overview'] = movies['overview'].apply(lambda x:x.split())

# =========================
# STEP 4: CREATE TAG COLUMN
# =========================

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']

# Convert list to string
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# New dataframe
new_df = movies[['movie_id','title','tags']]

# =========================
# STEP 5: TEXT VECTORIZATION
# =========================

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

# =========================
# STEP 6: COSINE SIMILARITY
# =========================

similarity = cosine_similarity(vectors)

# =========================
# STEP 7: RECOMMEND FUNCTION
# =========================

def recommend(movie):

    if movie not in new_df['title'].values:
        print("Movie not found")
        return

    index = new_df[new_df['title'] == movie].index[0]

    distances = similarity[index]

    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    print("\nRecommended movies:\n")

    for i in movie_list:
        print(new_df.iloc[i[0]].title)

# =========================
# STEP 8: TEST
# =========================

recommend("Batman Begins")