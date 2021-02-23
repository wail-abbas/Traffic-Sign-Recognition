from flask import Flask, request
from flask.templating import render_template
from sign_predictor import predict_sign
from tensorflow.keras.models import load_model


# load The Model
model_path = 'saved_models/model_sign.h5'
model = load_model(model_path)


# Run Flask website
app  = Flask("Traffic Sign Recognition")


@app.route('/result')
def get_image():
    html_form_data = dict(request.args)
    img = html_form_data['img']
    image = 'static/test_images/' + img
    result = predict_sign(image, model)
    return render_template('result.html', title='Traffic Sign Recognition', image = image, result = result)

@app.route('/')
def hello():
    return render_template('main.html', title='Traffic Sign Recognition')

if __name__ == "__main__":
    app.run(debug=True, port=5000)

