FROM python:3.10

RUN pip install --upgrade pip

# RUN pip install  git+https://github.com/huggingface/transformers sentencepiece datasets huggingsound
# RUN pip install flask 
RUN pip install -U openai-whisper



RUN apt-get update && apt-get install -y portaudio19-dev
RUN pip install --upgrade pip

RUN pip install flask ffmpeg-python
RUN pip install transformers sentencepiece datasets soundfile librosa huggingsound
RUN pip install torch torchaudio

WORKDIR /app

EXPOSE 5000

COPY . .

CMD [ "python", "app.py" ]

