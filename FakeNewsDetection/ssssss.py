

import pandas as pd
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import joblib



def add_news():

    try:
        text = input("Enter News: ")

        label = input("Enter Label (Real/Fake): ")

        with open("news.csv","a",newline="",encoding="utf-8") as file:

            writer = csv.writer(file)

            writer.writerow([text,label])

        print("News Added Successfully!")

    except Exception as e:

        print(e)

def view_news():
    
    df = pd.read_csv("news.csv")

    print(df)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def train_model():
    
    df = pd.read_csv("news.csv")

    X = df["Text"]
    y = df["Label"]

    vectorizer = TfidfVectorizer()

    X_vector = vectorizer.fit_transform(X)

    model = MultinomialNB()

    model.fit(X_vector, y)

    # Save model and vectorizer
    joblib.dump(model, "fake_news_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    print("Model Trained and Saved Successfully!")


def detect_news():
    
    try:

        # Load saved model
        model = joblib.load("fake_news_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")

        news = input("Enter News: ")

        news_vector = vectorizer.transform([news])

        prediction = model.predict(news_vector)

        probability = model.predict_proba(news_vector)

        confidence = max(probability[0]) * 100

        if prediction[0] == "Real":
            print("\n✅ News is REAL")
        else:
            print("\n❌ News is FAKE")

        print(f"Confidence Score: {confidence:.2f}%")

    except FileNotFoundError:
        print("Please train the model first (Option 3).")

    except Exception as e:
        print("Error:", e)
        
def menu():
    
    while True:

        print("\n===== Fake News Detection =====")
        print("1. Add News")
        print("2. View Dataset")
        print("3. Train Model")
        print("4. Detect News")
        print("5. Exit")

        choice = input("Enter Choice: ")

        if choice == "1":
            add_news()

        elif choice == "2":
            view_news()

        elif choice == "3":
            train_model()

        elif choice == "4":
            detect_news()

        elif choice == "5":
            print("Program Closed")
            break

        else:
            print("Invalid Choice!")

menu()
