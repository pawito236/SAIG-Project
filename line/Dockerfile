FROM python:3.10

RUN pip install --upgrade pip

# RUN pip install  git+https://github.com/huggingface/transformers sentencepiece datasets huggingsound


RUN pip install flask line-bot-sdk langchain openai tiktoken google-cloud-texttospeech

WORKDIR /app

EXPOSE 5000

ENV SERVER_URL="1"
ENV THIS_SERVER_URL="1"

COPY . .

CMD [ "python", "app.py" ]