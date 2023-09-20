import requests
import datetime
import errno
import json
import os
import sys
import tempfile
from argparse import ArgumentParser

from flask import Flask, request, abort, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model_tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from huggingsound import SpeechRecognitionModel
import torch

import whisper
model = whisper.load_model("base")

from flask import Flask, request, jsonify, send_file
app = Flask(__name__)

# @app.route('/stt', methods=['POST'])
# def upload_file():
#     uploaded_file = request.files['file']
#     uploaded_file.save('file.wav')
#     audio_paths = ['file.wav']
#     transcriptions = model_stt.transcribe(audio_paths)
#     return jsonify(transcriptions[0]["transcription"])

@app.route('/stt', methods=['POST'])
def upload_file():
    desired_languages = ['en', 'th']

    uploaded_file = request.files['file']
    uploaded_file.save('file.wav')
    audio = whisper.load_audio("file.wav")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")

    # Use a dictionary comprehension to filter the desired keys
    filtered_languages = {lang: prob for lang, prob in probs.items() if lang in desired_languages}

    # Find the language with the highest probability among 'en' and 'th'
    highest_probability_language = max(filtered_languages, key=filtered_languages.get)
    print(f"Detected language: ", highest_probability_language)

    # decode the audio
    options = whisper.DecodingOptions(language=highest_probability_language)
    result = whisper.decode(model, mel, options)
    return jsonify(result.text)

@app.route('/tts', methods=['POST'])
def tts():
    # start_time = time.time()
    data = request.get_data(as_text=True)

    inputs = processor(text=data, return_tensors="pt").to(device)
    speaker_embeddings = torch.tensor(embeddings_dataset[7200]["xvector"]).unsqueeze(0).to(device)
    speech = model_tts.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sf.write("speech.mp3", speech.cpu().numpy(), samplerate=16000)
    # print("Time for TTS ; ", time.time() - start_time)

    return send_file("speech.mp3", as_attachment=True)

@app.route('/status', methods=['POST'])
def status():
    data = request.get_data(as_text=True)
    return jsonify("ok")

if __name__ == "__main__":
    # arg_parser = ArgumentParser(
    #     usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    # )
    # arg_parser.add_argument('-p', '--port', type=int, default=5000, help='port')
    # arg_parser.add_argument('-d', '--debug', default=False, help='debug')
    # options = arg_parser.parse_args()

    # app.run(debug=options.debug, port=options.port)

    app.run(host='0.0.0.0', port=5000)

    