from flask import Flask, request
from flask import render_template
import predictor
app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('index.html')

#@app.route('/score/?title=<name>')
@app.route('/score/')
def hello(name="No title"):
    name=request.args.get('title', '')
    score=predictor.predict(str(name))
    return render_template('index.html', score=score)

app.run(debug=False)
