import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

# New student data
new_data = pd.DataFrame({
    "Attendance":[85],
    "StudyHours":[3],
    "PreviousMarks":[70],
    "Assignments":[80]
})

prediction = model.predict(new_data)

print("Predicted Final Marks:", prediction[0])