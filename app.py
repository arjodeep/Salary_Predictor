from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import os
import json
app = Flask(__name__)

# Load model parameters
with open("model_params.json", "r") as f:
    params = json.load(f)
    m = params["m"]
    b = params["b"]

def predict_salary(x):
    return m * x + b

def generate_plot(x_input=None, y_pred=None):
    # Dataset range
    X = list(range(0, 11))
    Y = [predict_salary(x) for x in X]

    plt.figure()
    plt.plot(X, Y, label="Regression Line", color="red")
    plt.scatter(X, Y, label="Training Data", color="black")

    if x_input is not None and y_pred is not None:
        plt.scatter([x_input], [y_pred], color='blue', label="Your Prediction", zorder=5)
    
    plt.xlabel("Years of Experience")
    plt.ylabel("Monthly Salary")
    plt.title("Salary Prediction Model")
    plt.legend()
    plt.tight_layout()
    
    # Save to static folder
    path = os.path.join("static", "plot.png")
    plt.savefig(path)
    plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_salary = None
    if request.method == "POST":
        try:
            experience = float(request.form["experience"])
            predicted_salary = predict_salary(experience)
            generate_plot(experience, predicted_salary)
        except:
            predicted_salary = "Invalid input"
    else:
        generate_plot()
    return render_template("index.html", predicted_salary=predicted_salary)
    
if __name__ == "__main__":
    app.run(debug=True)
