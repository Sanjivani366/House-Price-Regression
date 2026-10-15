import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model (Ensure your model file is renamed/saved as 'model.pkl')
with open('linear.pkl', 'rb') as f:
    model = pickle.load(f)

# Modern, responsive HTML template styled with Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-50 to-blue-50 min-h-screen font-sans text-slate-800">
    <div class="max-w-3xl mx-auto px-4 py-12">
        <!-- Header Section -->
        <div class="text-center mb-10">
            <span class="bg-blue-100 text-blue-800 text-xs font-semibold px-3 py-1 rounded-full uppercase tracking-wider">AI Regression Model</span>
            <h1 class="text-4xl font-extrabold tracking-tight text-slate-900 mt-2">House Price Prediction</h1>
            <p class="text-slate-600 mt-2">Enter the property attributes below to generate an instant market valuation.</p>
        </div>

        <!-- Form Card -->
        <div class="bg-white shadow-xl rounded-2xl p-8 border border-slate-100">
            <form action="/predict" method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Square Footage</label>
                    <input type="number" step="any" name="Square_Footage" required 
                        class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
                        placeholder="e.g. 2200">
                </div>

                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Number of Bedrooms</label>
                    <input type="number" name="Num_Bedrooms" required 
                        class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
                        placeholder="e.g. 3">
                </div>

                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Number of Bathrooms</label>
                    <input type="number" step="any" name="Num_Bathrooms" required 
                        class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
                        placeholder="e.g. 2">
                </div>

                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Year Built</label>
                    <input type="number" name="Year_Built" required 
                        class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
                        placeholder="e.g. 2012">
                </div>

                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Lot Size</label>
                    <input type="number" step="any" name="Lot_Size" required 
                        class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
                        placeholder="e.g. 4500">
                </div>

                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Garage Size (Cars)</label>
                    <input type="number" name="Garage_Size" required 
                        class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
                        placeholder="e.g. 2">
                </div>

                <div class="md:col-span-2">
                    <label class="block text-sm font-medium text-slate-700 mb-1">Neighborhood Quality (Scale 1–10)</label>
                    <input type="number" step="any" name="Neighborhood_Quality" required 
                        class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
                        placeholder="e.g. 8">
                </div>

                <div class="md:col-span-2 mt-4">
                    <button type="submit" 
                        class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg shadow-lg hover:shadow-xl transition duration-200">
                        Calculate Estimated Price
                    </button>
                </div>
            </form>

            {% if prediction_text %}
            <div class="mt-8 p-6 bg-emerald-50 border border-emerald-200 rounded-xl text-center">
                <h3 class="text-sm uppercase tracking-wide font-semibold text-emerald-800">Estimated Property Value</h3>
                <p class="text-3xl font-extrabold text-emerald-600 mt-2">{{ prediction_text }}</p>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract fields matching your model's exact expected feature order
        features = [
            float(request.form['Square_Footage']),
            float(request.form['Num_Bedrooms']),
            float(request.form['Num_Bathrooms']),
            float(request.form['Year_Built']),
            float(request.form['Lot_Size']),
            float(request.form['Garage_Size']),
            float(request.form['Neighborhood_Quality'])
        ]
        
        final_features = np.array([features])
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        
        return render_template_string(HTML_TEMPLATE, prediction_text=f"${output:,.2f}")
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error evaluating input: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
