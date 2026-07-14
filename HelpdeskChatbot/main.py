import pandas as pd

import csv
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def add_faq():

    try:

        question = input("Enter Question: ")
        answer = input("Enter Answer: ")
        intent = input("Enter Intent: ")

        with open("faq.csv", "a", newline="", encoding="utf-8") as file:

            writer = csv.writer(file)

            writer.writerow([question, answer, intent])

        print("FAQ Added Successfully!")

    except Exception as e:

        print("Error:", e)

def view_faq():
    
    df = pd.read_csv("faq.csv")

    print(df)

def train_chatbot():
    
    df = pd.read_csv("faq.csv")

    X = df["Question"]

    y = df["Intent"]

    vectorizer = TfidfVectorizer()

    X_vector = vectorizer.fit_transform(X)

    model = MultinomialNB()

    model.fit(X_vector, y)

    joblib.dump(model, "chatbot_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    print("Chatbot Trained Successfully!")

def ask_question():
    
    try:

        model = joblib.load("chatbot_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")

        df = pd.read_csv("faq.csv")

        question = input("Ask your question: ")

        question_vector = vectorizer.transform([question])

        intent = model.predict(question_vector)[0]

        answer = df[df["Intent"] == intent]["Answer"].iloc[0]

        print("\nChatbot:", answer)

    except FileNotFoundError:
        print("Please train the chatbot first.")

    except Exception as e:
        print("Error:", e)

def menu():
    
    while True:

        print("\n===== AI Helpdesk Chatbot =====")
        print("1. Add FAQ")
        print("2. View FAQ")
        print("3. Train Chatbot")
        print("4. Ask Question")
        print("5. Exit")

        choice = input("Enter Choice: ")

        if choice == "1":
            add_faq()

        elif choice == "2":
            view_faq()

        elif choice == "3":
            train_chatbot()

        elif choice == "4":
            ask_question()

        elif choice == "5":
            print("Program Closed")
            break

        else:
            print("Invalid Choice!")

menu()
     
