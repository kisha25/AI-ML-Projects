from sklearn.linear_model import LinearRegression


import pandas as pd
import csv
import pandas as pd
import csv

def add_data():

    try:
        month = int(input("Enter Month: "))
        usage = float(input("Enter Usage: "))

        with open("utility_data.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([month, usage])

        print("Data Added Successfully!")

    except ValueError:
        print("Please enter valid numbers.")


def view_data():
    
    try:
        df = pd.read_csv("utility_data.csv")
        print(df)

    except FileNotFoundError:
        print("CSV file not found.")
        
def update_data():
    try:
        df = pd.read_csv("utility_data.csv")

        month = int(input("Enter Month to Update: "))
        new_usage = float(input("Enter New Usage: "))

        if month in df["Month"].values:
            df.loc[df["Month"] == month, "Usage"] = new_usage

            df.to_csv("utility_data.csv", index=False)

            print("Data Updated Successfully!")
        else:
            print("Month not found!")

    except Exception as e:
        print("Error:", e)


def remove_data():
    try:
        df = pd.read_csv("utility_data.csv")

        month = int(input("Enter Month to Remove: "))

        if month in df["Month"].values:

            df = df[df["Month"] != month]

            df.to_csv("utility_data.csv", index=False)

            print("Data Removed Successfully!")

        else:
            print("Month not found!")

    except Exception as e:
        print("Error:", e)

    
def train_model():
    
    df = pd.read_csv("utility_data.csv")

    X = df[["Month"]]

    y = df["Usage"]

    model = LinearRegression()

    model.fit(X, y)

    return model

def predict_usage():
    
    try:

        model = train_model()

        future_month = int(input("Enter Future Month: "))

        prediction = model.predict(
            pd.DataFrame([[future_month]], columns=["Month"])
)
        print(f"Predicted Usage: {prediction[0]:.2f}")

    except Exception as e:
        print("Error:", e)

def menu():
    
    while True:

        print("\n===== Utility Usage Prediction =====")
        print("1. Add Data")
        print("2. View Data")
        print("3. Update Data")
        print("4. Remove Data")
        print("5. Predict Usage")
        print("6. Exit")
 
        

        choice = input("Enter Choice: ")

        if choice == "1":
            add_data()
            
        elif choice == "2":
            view_data() 

        elif choice == "3":
            update_data()

        elif choice == "4":
            remove_data()

        elif choice == "5":
            predict_usage()

        elif choice == "6":
            print("Program Closed")
            break

     
        else:
            print("Invalid Choice")

menu()         
 