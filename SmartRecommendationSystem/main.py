import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_model():
    
    df = pd.read_csv("movies.csv")

    vectorizer = TfidfVectorizer()

    genre_matrix = vectorizer.fit_transform(df["Genre"])

    similarity = cosine_similarity(genre_matrix)

    return df, similarity

def recommend_movies():
    
    df = pd.read_csv("movies.csv")

    movie_name = input("Enter Movie Name: ")

    if movie_name not in df["Movie"].values:
        print("Movie not found.")
        return

    vectorizer = TfidfVectorizer()

    genre_matrix = vectorizer.fit_transform(df["Genre"])

    similarity = cosine_similarity(genre_matrix)

    movie_index = df[df["Movie"] == movie_name].index[0]

    scores = list(enumerate(similarity[movie_index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    print("\nRecommended Movies:\n")

    for i in scores[1:6]:

        print(
            df.iloc[i[0]]["Movie"],
            "- Similarity:",
            round(i[1] * 100, 2),
            "%"
        )
            
def menu():
    
    while True:

        print("\n===== Smart Recommendation System =====")
        print("1. View Movies")
        print("2. Get Recommendations")
        print("3. Exit")

        choice = input("Enter Choice: ")

        if choice == "1":

            df = pd.read_csv("movies.csv")
            print(df)

        elif choice == "2":

            recommend_movies()

        elif choice == "3":

            print("Program Closed")
            break

        else:

            print("Invalid Choice!")
            
menu()
