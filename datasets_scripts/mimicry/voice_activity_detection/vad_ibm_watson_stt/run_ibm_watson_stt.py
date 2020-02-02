#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Call the IBM Watson speech-to-text service and save the outputs in json format. 
#     Applied to Mimicry dataset to determine voice activity.
#######################################################################################################################


import json
import glob
import os
from ibm_watson import SpeechToTextV1


service = SpeechToTextV1(
    url='https://stream.watsonplatform.net/speech-to-text/api',
    # Janko's
    # iam_apikey='7CgbkXI9nAf2rRfMdmiVF1rU2CQKzpcB32TReUim6wAo'
    # Kalin's
    iam_apikey='KnrKhnPyfgD_BjrKoTFXtvJ8BhVj_u8_8w5jIypAKznK'
)

mono_audio_dir = '/home/ICT2000/jondras/dvra_datasets/mimicry/audio/audio_separated_8kHz'
stt_outputs_dir = f'/home/ICT2000/jondras/dvra_datasets/mimicry/voice_activity_detection/speech_to_text_ibm_watson'
if not os.path.exists(stt_outputs_dir):
    os.makedirs(stt_outputs_dir)

# Iterate over mono audio inputs
for i, audio_filename in enumerate(sorted(glob.glob(f'{mono_audio_dir}/*.wav'))):

    with open(audio_filename, 'rb') as audio_file:
        response = service.recognize(audio=audio_file, 
                                     content_type='audio/wav', 
                                     # Mimicry DB comes from the UK
                                     model='en-GB_NarrowbandModel', 
                                     timestamps=True, 
                                     word_confidence=True, 
                                     speaker_labels=True
                                    ).get_result()

        # Save the response as json
        audio_basename = audio_filename.split('/')[-1][:-4]
        with open(f'{stt_outputs_dir}/stt_{audio_basename}.json', "w") as json_output_file:
            json.dump(response, json_output_file, indent=4)
            
    print(f'{i + 1}\t{audio_basename}\t{audio_filename}')

print(f'\nSaved {i + 1} outputs from IBM Watson speech-to-text to {stt_outputs_dir}.')
