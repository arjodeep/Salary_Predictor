import pandas as pd
import matplotlib.pyplot as plt
import json
data=pd.read_csv('Salary_Data.csv')

# def loss_function(m, b, points):
#     total_error=0
#     for i in range(len(points)):
#         x=points.iloc[i].YearsExperience
#         y=points.iloc[i].Salary
#         total_error+=(y-(m*x+b))**2
#     total_error/float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient=0
    b_gradient=0
    n=len(points)
    for i in range(n):
        x=points.iloc[i].YearsExperience
        y=points.iloc[i].Salary
        m_gradient+=-(2/n)*x*(y-(m_now*x+b_now))
        b_gradient+=-(2/n)*(y-(m_now*x+b_now))
    m=m_now - m_gradient*L
    b=b_now - b_gradient*L
    return m,b

m=0
b=0
L=0.0001
epochs=2000

for i in range(epochs):
    m,b=gradient_descent(m, b, data, L)


print(m, b)
# plt.scatter(data.YearsExperience, data.Salary, color="black")
# plt.plot(list(range(1,11)), [m*x + b for x in range(1,11)], color="red")
# plt.show()

# Example input
years_experience = float(input("Enter years of experience: "))

# Prediction using learned model
predicted_salary = m * years_experience + b

print(f"Predicted Salary: ₹{predicted_salary:.2f}")
# Save model parameters
model_params = {"m": m, "b": b}
with open("model_params.json", "w") as f:
    json.dump(model_params, f)

