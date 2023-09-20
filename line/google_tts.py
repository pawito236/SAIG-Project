import os

# https://cloud.google.com/text-to-speech/docs/voices

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="plant-pot-qjbt-e6beedd4d5e3.json"

from google.cloud import texttospeech

# Instantiates a client
client = texttospeech.TextToSpeechClient()

# Set the text input to be synthesized
# synthesis_input = texttospeech.SynthesisInput(text="Ok, let me finish this point first.")
synthesis_input = texttospeech.SynthesisInput(text="สวัสดีค่ะ มีอะไรให้ช่วยไหมคะ")

# Build the voice request, select the language code ("en-US") 
# ****** the NAME
# and the ssml voice gender ("neutral")
# voice = texttospeech.VoiceSelectionParams(
#     language_code='en-US',
#     # name='en-US-Wavenet-D',
#     name = 'en-US-Standard-D',
#     ssml_gender=texttospeech.SsmlVoiceGender.MALE)

voice = texttospeech.VoiceSelectionParams(
    language_code='th-TH',
    # name='en-US-Wavenet-D',
    name = 'th-TH-Neural2-C',
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)

# Select the type of audio file you want returned
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
# response = client.synthesize_speech(synthesis_input, voice, audio_config)
response = client.synthesize_speech(
    input=synthesis_input,
    voice=voice,
    audio_config=audio_config
)
# The response's audio_content is binary.
with open('greeting_sopracie1.mp3', 'wb+') as out:
    # Write the response to the output file.
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')