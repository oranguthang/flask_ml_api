import flask
from catboost import CatBoostClassifier
from flask import request, jsonify

app = flask.Flask(__name__)
app.config['DEBUG'] = False


@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        json = request.json

        model = CatBoostClassifier()  # parameters not required.
        model.load_model(f'models/{model_name}/model')
        predicted = model.predict(json)

        return jsonify({'predictions': predicted.tolist()})
    else:
        return 'Content-Type not supported!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
