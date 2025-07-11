from flask import Flask, render_template, request
import joblib
import numpy as np

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load humidity prediction model
model = joblib.load('rf_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        t_degc = float(request.form['t_degc'])
        wv = float(request.form['wv_m/s'])
        tdew = float(request.form['tdew_degc'])

        input_data = np.array([[t_degc, wv, tdew]])
        prediction = model.predict(input_data)[0]

        # Decide on GIF
        if prediction >= 70:
            gif_url = "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExcHNtZjc3OTN4ang0bmx1c2ZsYWs1Z2xqMnF4MGJlZWprd2J1YXVkbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/uXvkMOSnH2QUO2JH9e/giphy.gif"
        else:
            gif_url = "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExOTQ5dmczbHJtYjd0bDZuaWhicXR6eWZnY2dkMG9meThkMnV5d3R1aiZlcD12MV9zdGlja2Vyc19zZWFyY2gmY3Q9cw/lqSArC8vsDTuXvTR3i/200.webp"

        return render_template(
            'index.html',
            result=f"{prediction:.2f}%",
            gif_url=gif_url
        )

    except Exception as e:
        return render_template('index.html', result="Invalid input. Please enter numbers only.", gif_url=None)

if __name__ == '__main__':
    app.run(debug=True)

