import azure.cognitiveservices.speech as speechsdk
import os
from dotenv import load_dotenv

load_dotenv()

speech_key = os.getenv("Speechkey")
speech_endpoint = os.getenv("speech_endpoint")

speech_config = speechsdk.SpeechConfig(
    subscription=speech_key,
    endpoint=speech_endpoint
)



def azure_ipa(text, ipa, language, synthesis_voice_name, output_file,
              rate="0%", volume="0%", pitch="0%", pause_ms="500ms", speech_config=speech_config):


    speech_config.speech_synthesis_voice_name = synthesis_voice_name

    audio_config = speechsdk.audio.AudioOutputConfig(
        filename=output_file
    )

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    # Allow single sentence or multiple sentences
    if isinstance(text, str):
        text = [text]

    if isinstance(ipa, str):
        ipa = [ipa]

    if len(text) != len(ipa):
        raise ValueError(
            "text and ipa must have the same number of sentences"
        )

    phoneme_blocks = ""

    for t, p in zip(text, ipa):

        phoneme_blocks += f"""
        <phoneme alphabet="ipa" ph="{p}">
            {t}
        </phoneme>

        <break time="{pause_ms}"/>
        """

    ssml = f"""
    <speak version="1.0"
        xmlns="http://www.w3.org/2001/10/synthesis"
        xml:lang="{language}">

        <voice name="{synthesis_voice_name}">

            <prosody rate="{rate}"
                    volume="{volume}"
                    pitch="{pitch}">

                {phoneme_blocks}

            </prosody>

        </voice>

    </speak>
    """

    result = synthesizer.speak_ssml_async(ssml).get()

    # if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    #     print(f"Created: {output_file}")
    #     return True

    # elif result.reason == speechsdk.ResultReason.Canceled:

    #     details = result.cancellation_details

    #     print("Speech synthesis canceled")
    #     print("Reason:", details.reason)

    #     if details.reason == speechsdk.CancellationReason.Error:
    #         print("Error:", details.error_details)

    #     return False
    
def azure_text(text, language, synthesis_voice_name, output_file,
               rate="0%", volume="0%", pitch="0%", speech_config=speech_config):

    speech_config.speech_synthesis_voice_name = synthesis_voice_name

    audio_config = speechsdk.audio.AudioOutputConfig(
        filename=output_file
    )

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    ssml = f"""
    <speak version="1.0"
        xmlns="http://www.w3.org/2001/10/synthesis"
        xml:lang="{language}">

        <voice name="{synthesis_voice_name}">

            <prosody rate="{rate}"
                    volume="{volume}"
                    pitch="{pitch}">

                {text}

            </prosody>

        </voice>

    </speak>
    """

    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Created: {output_file}")
        return True

    else:
        print("Synthesis failed")
        print(result.cancellation_details.error_details)
        return False