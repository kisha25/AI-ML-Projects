import matplotlib.pyplot as plt
import pandas as pd
import csv

import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


print("Current Folder:", os.getcwd())
print("CSV File:", os.path.abspath("student_data.csv"))

def add_student():

    try:
        attendance = int(input("Enter Attendance (%): "))
        marks = int(input("Enter Marks: "))
        study_hours = float(input("Enter Study Hours: "))
        final_performance = int(input("Enter Final Performance: "))

        with open("student_data.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                attendance,
                marks,
                study_hours,
                final_performance
            ])

        print("Student Added Successfully!")

    except ValueError:
        print("Please enter valid numbers.")


def view_students():

    try:
        df = pd.read_csv("student_data.csv")

        print("\n===== Student Records =====")
        print(df)

    except FileNotFoundError:
        print("student_data.csv file not found.")


def delete_student():
    try:
        df = pd.read_csv("student_data.csv")

        print(df)

        attendance = int(input("Enter Attendance to Delete: "))

        if attendance in df["Attendance"].values:

            df = df[df["Attendance"] != attendance]

            df.to_csv("student_data.csv", index=False)

            print("Student Deleted Successfully!")

            # Check if it was saved
            print("\nUpdated Data:")
            print(pd.read_csv("student_data.csv"))

        else:
            print("Attendance not found!")

    except Exception as e:
        print("Error:", e)

from sklearn.linear_model import LinearRegression

def train_model():

    df = pd.read_csv("student_data.csv")

    # Input features
    X = df[["Attendance", "Marks", "StudyHours"]]

    # Output (Target)
    y = df["FinalPerformance"]

    # Create model
    model = LinearRegression()

    # Train model
    model.fit(X, y)

    return model

def predict_performance():
    
    try:

        model = train_model()

        attendance = int(input("Enter Attendance: "))
        marks = int(input("Enter Marks: "))
        study_hours = float(input("Enter Study Hours: "))

        future_student = pd.DataFrame(
            [[attendance, marks, study_hours]],
            columns=["Attendance", "Marks", "StudyHours"]
        )

        prediction = model.predict(future_student)

        print(f"Predicted Final Performance: {prediction[0]:.2f}")

    except Exception as e:
        print("Error:", e)

def evaluate_model():
    
    df = pd.read_csv("student_data.csv")

    X = df[["Attendance", "Marks", "StudyHours"]]
    y = df["FinalPerformance"]

    # Split the data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = r2_score(y_test, predictions)

    print(f"\nModel Accuracy (R² Score): {accuracy:.2f}")

def show_chart():
    
    df = pd.read_csv("student_data.csv")

    plt.figure(figsize=(8,5))

    plt.plot(df["Attendance"], label="Attendance", marker="o")
    plt.plot(df["Marks"], label="Marks", marker="s")
    plt.plot(df["FinalPerformance"], label="Final Performance", marker="^")

    plt.title("Student Performance Analysis")
    plt.xlabel("Student Number")
    plt.ylabel("Value")

    plt.legend()
    plt.grid(True)

    plt.show()
    
    
def menu():
    
    while True:

        print("\n===== Student Performance Prediction =====")
        print("1. Add Student")
        print("2. View Students")
        print("3. Delete Student")
        print("4. Predict Performance")
        print("5. Evaluate Model")
        print("6. Show Chart")
        print("7. Exit")

        choice = input("Enter Choice: ")

        if choice == "1":
            add_student()

        elif choice == "2":
            view_students()

        elif choice == "3":
            delete_student()

        elif choice == "4":
            predict_performance()

        elif choice == "5":
            evaluate_model()

        elif choice == "6":
            show_chart()

        elif choice == "7":
            print("Program Closed")
            break

        else:
            print("Invalid Choice!")

menu()