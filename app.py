from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form values to float and check for negative values
        features = [float(x) for x in request.form.values()]
        if any(f < 0 for f in features):
            prediction_text = 'Không được nhập số âm.'
        else:
            final_features = np.array(features).reshape(1, -1)
            
            # Predict
            prediction = model.predict(final_features)[0]
            
            # Check if the prediction is less than 0
            if prediction < 0:
                prediction_text = 'Dự đoán không hợp lệ (giá nhà không thể âm).'
            else:
                prediction_text = 'Giá Nhà Dự Đoán: ${:.2f}'.format(prediction)
    
    except ValueError:
        prediction_text = 'Vui lòng nhập đúng định dạng số.'
    except Exception as e:
        prediction_text = f'Đã xảy ra lỗi: {str(e)}'

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
