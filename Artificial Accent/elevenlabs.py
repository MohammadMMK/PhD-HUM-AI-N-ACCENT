from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import os

load_dotenv()

elevenlabs = ElevenLabs(
  api_key= os.getenv("ELEVENLABS_API_KEY")
)

audio = elevenlabs.text_to_speech.convert(
    text= '"/ðə/" red "/θif/" stole "/θɹi/" "/θɪŋz/".',
    voice_id="EST9Ui6982FZPSi7gCHi",  # "English standard" - browse voices at elevenlabs.io/app/voice-library
    model_id="eleven_v3",
    output_format="mp3_44100_128"
)

play(audio)

