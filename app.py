import keras
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from utils.findTriple import train_model
from utils.findTriple import getTriple

sess=keras.backend.get_session()
graph=tf.get_default_graph()
train_model.load_weights('F:/demo/relation_extraction-Demo/utils/save/best_model.weights')


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('Demo.html')

@app.route('/IEDemo', methods=['GET', 'POST'])
def IEDemo():
    text = str(request.args.get("text"))
    # 此处写flask调用keras的代码逻辑
    with sess.as_default():
        with graph.as_default():
            ieData = getTriple(text)
    return jsonify(ieData)


if __name__ == '__main__':
    app.debug = True
    app.run("0.0.0.0")#localhost:5000/
