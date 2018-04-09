import os
from flask import Flask, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from model import *
from inference.Compiler import *

UPLOAD_FOLDER = '/app/uploads'
ALLOWED_EXTENSIONS = set([
    'png'
    ])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.debug = True


@app.route('/ping')
def hello():
    return 'Hello world!'


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # check if the post request has the file part

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # if user does not select file, browser also
        # submit a empty part without filename

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],
                      filename))
            encoder, decoder = load_model()
            star_text = '<START>'
            hidden = decoder.init_hidden()
            image = resize_img(os.path.join(app.config['UPLOAD_FOLDER'],
                      filename))
            image = Variable(torch.FloatTensor([image]))
            predicted = '<START> '
            for di in range(9999):
                sequence = id_for_word(star_text)
                decoder_input = Variable(torch.LongTensor([sequence])).view(1,-1)
                features = encoder(image)
                outputs,hidden = decoder(features, decoder_input,hidden)
                topv, topi = outputs.data.topk(1)
                ni = topi[0][0][0]
                word = word_for_id(ni)
                if word is None:
                        continue
                predicted += word + ' '
                star_text = word
                print(predicted)
                if word == '<END>':
                        break
            compiler = Compiler('default')
            compiled_website = compiler.compile(predicted.split())

            return compiled_website
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
def load_model():
    encoder = torch.load('model_weights/encoder_resnet34_0.061650436371564865.pt')
    decoder = torch.load('model_weights/decoder_resnet34_0.061650436371564865.pt')
    return encoder, decoder

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() \
        in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)