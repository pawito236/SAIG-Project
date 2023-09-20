import requests
import datetime
import errno
import json
import os
import sys
import tempfile
from argparse import ArgumentParser
import re

from flask import Flask, request, abort, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    LineBotApiError, InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    TemplateSendMessage, ConfirmTemplate, MessageAction,
    ButtonsTemplate, ImageCarouselTemplate, ImageCarouselColumn, URIAction,
    PostbackAction, DatetimePickerAction,
    CameraAction, CameraRollAction, LocationAction,
    CarouselTemplate, CarouselColumn, PostbackEvent,AudioSendMessage,
    StickerMessage, StickerSendMessage, LocationMessage, LocationSendMessage,
    ImageMessage, VideoMessage, AudioMessage, FileMessage,
    UnfollowEvent, FollowEvent, JoinEvent, LeaveEvent, BeaconEvent,
    MemberJoinedEvent, MemberLeftEvent, UnknownEvent,
    FlexSendMessage, BubbleContainer, ImageComponent, BoxComponent,
    TextComponent, IconComponent, ButtonComponent,
    SeparatorComponent, QuickReply, QuickReplyButton,
    ImageSendMessage)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import os

import sys
from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.schema.messages import BaseMessage
from google.cloud import texttospeech
# import pinecone

from google.cloud import firestore
from google.oauth2 import service_account

import os
# Set the path to your service account JSON file
service_account_path = "plant-pot-qjbt-e6beedd4d5e3.json"

# Set the environment variable for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path

# Initialize Firestore client
credentials = service_account.Credentials.from_service_account_file(service_account_path)
db = firestore.Client(credentials=credentials)

client = texttospeech.TextToSpeechClient()
voice = texttospeech.VoiceSelectionParams(
    language_code='th-TH',
    # name='en-US-Wavenet-D',
    name = 'th-TH-Neural2-C',
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3)



server_url = os.environ.get('SERVER_URL') if os.environ.get('SERVER_URL') != None else "http://a87e-34-87-10-20.ngrok-free.app"
this_server_url = os.environ.get('THIS_SERVER_URL') if os.environ.get('THIS_SERVER_URL') != None else "https://45ee-2001-fb1-a8-c558-89b2-b32-7e51-e926.ngrok-free.app"

with open('keys.json', 'r') as json_file:
    data_keys = json.load(json_file)

os.environ["OPENAI_API_KEY"] = data_keys["OPENAI_API_KEY"]
global memory_global
memory_global = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=3)

app = Flask(__name__, static_folder="static")
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)

# get channel_secret and channel_access_token from your environment variable
channel_secret = data_keys["channel_secret"]
channel_access_token = data_keys["channel_access_token"]
if channel_secret is None or channel_access_token is None:
    print('Specify LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN as environment variables.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')

# function for create tmp dir for download content
def make_static_tmp_dir():
    try:
        os.makedirs(static_tmp_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(static_tmp_path):
            pass
        else:
            raise


def create_user_persona(user_id):

    doc_ref = db.collection('SAIG-Project').document(user_id)
    doc = doc_ref.get()
    if not doc.exists:
        doc_ref.set({
            "name": "",
            "status_type" : "",
            "status_water" : "", # low, well
            "status_light" : "", # low, well
            "q" : [],
            "a" : []
        })  

    print("Done init persona")

def get_user_document(user_id):
    doc_ref = db.collection('SAIG-Project').document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        user_dict = doc.to_dict()
        return user_dict
    else:
        create_user_persona(user_id)
        doc_ref = db.collection('SAIG-Project').document(user_id)
        doc = doc_ref.get()
        user_dict = doc.to_dict()
        return user_dict

def update_user_data(user_id, data):
    if(len(data) > 0):
        doc_ref = db.collection('SAIG-Project').document(user_id)
        doc_ref.update(data)

def reset_user_persona(user_id):

    doc_ref = db.collection('SAIG-Project').document(user_id)
    doc_ref.set({
        "name": "",
    })  

    print("Done reset persona")


def check_lang_code(recog_text):
        
        # Thai
    pattern = re.compile(r'[\u0E00-\u0E7F]')
    if(bool(pattern.search(recog_text))):
        print("Found Thai")
        return "th-th"
        
    # other is eng
    else:
        return "en-us"


def custom_langchain_stream(question, user_dict):
  
    memory_local = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=4)
    for i in range(len(user_dict['a'])):
        memory_local.save_context(user_dict['q'][i], user_dict['a'][i])

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"Act like a little girl named lily. Reply with short Answer the following question as cute as possible."), # The persistent system prompt
        MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injected
    ])

    llm = ChatOpenAI(model='gpt-3.5-turbo', streaming=True)

    chat_llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory_local,
    )

    return chat_llm_chain.predict(human_input=question)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except LineBotApiError as e:
        print("Got exception from LINE Messaging API: %s\n" % e.message)
        for m in e.error.details:
            print("  %s: %s" % (m.property, m.message))
        print("\n")
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    question = event.message.text
    user_id = event.source.user_id

    create_user_persona(user_id)
    user_dict = get_user_document(user_id)

    print(user_dict)

    # for i in range(len(user_dict['a'])):
    #     memory_global.save_context(user_dict['q'][i], user_dict['a'][i])

    ans = custom_langchain_stream(question, user_dict)
    line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=ans))
    
    update_dict = {
        "q" : user_dict['q'][-5:] + [{"input" : question}],
        "a" : user_dict['a'][-5:] + [{"output" : ans}]
    }

    update_user_data(user_id, update_dict)
            
@handler.add(MessageEvent, message=(ImageMessage, VideoMessage, AudioMessage))
def handle_content_message(event):
    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
    elif isinstance(event.message, VideoMessage):
        ext = 'mp4'
    elif isinstance(event.message, AudioMessage):
        # ext = 'm4a'
        ext = 'wav'
    else:
        return
    
    user_id = event.source.user_id

    create_user_persona(user_id)
    user_dict = get_user_document(user_id)

    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix=ext + '-', delete=False) as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        tempfile_path = tf.name

    dist_path = tempfile_path + '.' + ext
    dist_name = os.path.basename(dist_path)
    os.rename(tempfile_path, dist_path)
    
    with open(dist_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(server_url + "/stt", files=files)
        
    if response.status_code == 200:
        print(f'File {dist_path} stted successfully.')
        stt_respond = json.loads(response.text)
        print("Yu say : ", stt_respond)
    else:
        stt_respond = ""
    
    ans = custom_langchain_stream(stt_respond, user_dict)
    print("GPT : ", ans)

    template_answer = "You : {}\n\nMe : {}".format(stt_respond, ans)

    rec_file = "static/tmp/audio_file{}.wav".format(len(os.listdir("static/tmp")))

    lang = check_lang_code(ans)

    if(lang == "en-us"):
        response_tts = requests.post(server_url + "/tts", data=ans.encode())
        if response_tts.status_code == 200:
            with open(rec_file, 'wb') as file:
                file.write(response_tts.content)
                print("Adding file to queue")
        else:
            print("Error TTS.")
    else:
        # th-th
        synthesis_input = texttospeech.SynthesisInput(text=ans)
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        # The response's audio_content is binary.
        with open(rec_file, 'wb+') as out:
            # Write the response to the output file.
            out.write(response.audio_content)

    print("Generate payload audio")

    audio_message = AudioSendMessage(
        original_content_url= this_server_url+"/{}".format(rec_file),
        duration=5000  # Duration in milliseconds (60 seconds in this example)
    )
    line_bot_api.reply_message(
                    event.reply_token,
                    [TextSendMessage(text=template_answer),
                     audio_message])
    
    update_dict = {
        "q" : user_dict['q'][-5:] + [{"input" : stt_respond}],
        "a" : user_dict['a'][-5:] + [{"output" : ans}]
    }

    update_user_data(user_id, update_dict)

if __name__ == "__main__":
    # arg_parser = ArgumentParser(
    #     usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    # )
    # arg_parser.add_argument('-p', '--port', type=int, default=5000, help='port')
    # arg_parser.add_argument('-d', '--debug', default=True, help='debug')
    # options = arg_parser.parse_args()
    # app.run(debug=options.debug, port=options.port)

    # create tmp dir for download content
    make_static_tmp_dir()

    
    app.run(host='0.0.0.0', port=5000)