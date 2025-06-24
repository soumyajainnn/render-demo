from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)
df_columns = ['Sale Price', 'Mrp', 'Discount Percentage', 'Ram']

@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = 'Top Seller' if prediction[0] == 1 else 'Not a Top Seller'
    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == '__main__':
    app.run(debug=True)