import json
from operator import imod
from flask import Flask, render_template, request
from findTraits import findTraits
from flask import jsonify
from getPersonalityType import run 

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['formInput']
    per,some = run(text)
    return render_template('result.html',personalityType=per,jsonData=(findTraits(text)))

if __name__ == '__main__':
  app.run(debug=True)