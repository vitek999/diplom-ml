from flask import Flask, request

from test_saved_model_use import predict_clazz_exercise

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.post('/predict')
def login_post():
    print(request.json)
    clazz = predict_clazz_exercise(request.json)
    return {
        "result": clazz
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)