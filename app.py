from flask import Flask,render_template, request
from src.predict import predict

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_news():
    text = request.form["news_text"]
    result = predict(text)  ## storing result of predict.py here
    return render_template("index.html", result=result, news_text=text)

if __name__ == "__main__":
    app.run(debug=True)