import os
import logging
import subprocess
import nemo.collections.asr as nemo_asr
import json
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME
import requests

def update_task(task_id, updated_data, label_studio_url, api_key):
    update_url = f"{label_studio_url}/api/tasks/{task_id}"
    headers = {
        'Authorization': f'Token {api_key}'
    }
    response = requests.patch(update_url, json=updated_data, headers=headers)
    return response.json()

# Example usage
label_studio_url = 'http://host.docker.internal:8080'  # Replace with your Label Studio URL
api_key = 'f40ee9f855f4847efb07cde32b8057bfe32ce74e'  # Your API token


logger = logging.getLogger(__name__)

class NemoASR(LabelStudioMLBase):
    def __init__(self, model_name='stt_en_fastconformer_hybrid_large_pc', **kwargs):
        super(NemoASR, self).__init__(**kwargs)
        self.processed_audio_path=[]
        self.model_name_base='nvidia/stt_en_fastconformer_transducer_large'
        # Binding ASR model to TextArea control tag
        self.from_name, self.to_name, self.value = self._bind_to_textarea()

        # Initialize NeMo ASR model
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name)
    #stt_en_fastconformer_hybrid_large_pc
    #nvidia/stt_en_fastconformer_transducer_large
#    def process_ctm_file(self, ctm_file_path):
#        annotations = []
#        current_annotation = {"start": None, "text": "", "author": "author"}
#        segment_duration = 10.0  # Target duration in seconds for each segment#
#
#        with open(ctm_file_path, 'r') as file:
#            for line in file:
#                parts = line.strip().split()
#                if len(parts) >= 4:
#                    name, channel, start_time, duration, word = parts
#                    start_time = float(start_time)
#                    end_time = start_time + float(duration)
#
#                    if current_annotation["start"] is None:
#                        current_annotation["start"] = start_time
#
#                    if current_annotation["text"] and (end_time - current_annotation["start"] > segment_duration):
#                        
#                        current_annotation["end"] = previous_end_time
#                        annotations.append(current_annotation)
#                        
#                        current_annotation = {"start": start_time, "text": "", "author": "author"}
#
 #                   current_annotation["text"] += word + " "
  #                  previous_end_time = end_time
#
 #           # Add the last annotation if it exists
  #          if current_annotation["text"]:
   #             current_annotation["end"] = previous_end_time
    #            annotations.append(current_annotation)
#
 #       return annotations

    def process_ctm_file(self, ctm_file_path):
        annotations = []
        current_annotation = {"start": None, "text": "", "author": "author"}
        segment_duration_min = 10.0  # Minimum target duration in seconds for each segment
        segment_duration_max = 20.0  # Maximum duration in seconds for each segment

        with open(ctm_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, _, start_time, duration, word = parts
                    start_time = float(start_time)
                    end_time = start_time + float(duration)

                    if current_annotation["start"] is None:
                        current_annotation["start"] = start_time

                    is_end_of_sentence = word.endswith(('.', '?', '!'))

                    # If the annotation has not started or the word is part of a new sentence that fits within a new segment
                    if current_annotation["start"] is not None:
                        current_duration = start_time - current_annotation["start"]

                    if current_annotation["text"] and (is_end_of_sentence or current_duration >= segment_duration_max):
                        # If the current segment exceeds max duration or if the word ends a sentence
                        if current_duration >= segment_duration_min or is_end_of_sentence:
                            current_annotation["text"] += word + " "
                            current_annotation["end"] = end_time
                            annotations.append(current_annotation)
                            current_annotation = {"start": None, "text": "", "author": self.model_name_base, "end": None}
                            continue
                        else:
                            # Adjust current_annotation to not include the last word if it's not ending with a sentence marker
                            # and the duration is less than the minimum (handled by appending before this condition)
                            current_annotation["end"] = previous_end_time

                    if current_annotation["start"] is None:
                        current_annotation["start"] = start_time

                    current_annotation["text"] += word + " "
                    previous_end_time = end_time  # Update previous_end_time after adding the word

            # Ensure the last annotation is added if it contains text and hasn't been added yet
            if current_annotation["text"]:
                if "end" not in current_annotation or current_annotation["end"] is None:
                    current_annotation["end"] = previous_end_time
                annotations.append(current_annotation)

        return annotations



	


    def predict(self, tasks, model_name='stt_en_fastconformer_hybrid_large_pc',  **kwargs):
        for task in tasks:
            relative_audio_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
            full_audio_url = f"http://host.docker.internal:8080{relative_audio_url}"
            local_audio_filename = os.path.basename(relative_audio_url)
            local_audio_path = os.path.join('/tmp', local_audio_filename)

            # Downloading the audio file
            if local_audio_path not in self.processed_audio_path:
                self.processed_audio_path.append(local_audio_path)
            else:
                return tasks
            
            try:
                subprocess.run(['wget', '-O', local_audio_path, full_audio_url], check=True)

            except subprocess.CalledProcessError as e:
                logger.error(f"Error downloading audio file: {e}")
                continue
            
            # Transcribe audio file
            transcription = self.model.transcribe(paths2audio_files=[local_audio_path])[0][0]
            

            
            # Initialize NemoForcedAligner (adjust model and device as needed)
            #aligner = NemoForcedAligner(model='nvidia/stt_en_fastconformer_transducer_large', device='cuda')

            # Perform forced alignment
            #alignment_result = aligner.align(local_audio_path, transcription)

            # Update task with alignment result
            #task['data']['alignment'] = alignment_result
            WORK_DIR="./"
            manifest_filepath = f"{WORK_DIR}manifest.json"
            manifest_data = {
                "audio_filepath": local_audio_path,
                "text": transcription
                            }
            with open(manifest_filepath, 'w') as f:
                line = json.dumps(manifest_data)
                f.write(line + "\n")
                
            nfa_output_path = "/app/nfaout/" 
            NFA_PATH = "./"  # Replace with your NeMo installation path
            subprocess.run([
                     f"/app/align.py",
                     f"pretrained_name=stt_en_fastconformer_hybrid_large_pc",
                     f"manifest_filepath={manifest_filepath}",
                     f"output_dir={nfa_output_path}",
                     "additional_segment_grouping_separator=|",
                     ], check=True)
            
            # Path to the NFA output JSON file
            #3dda4563-7850-286674-0001.wav
            #/app/nfaout/ctm/words/3dda4563-7850-286674-0001.ctm
            #local_audio_filename
            # Read NFA output
            npath=str(nfa_output_path+'ctm'+'/words/'+local_audio_filename[:]).replace('.wav', '.ctm')
            ctm_words = self.process_ctm_file(npath)
            

            # with open(str(nfa_output_path+'ctm'+'/words/'+local_audio_filename[:]).replace('.wav', '.ctm'), 'r') as file:
                # ctm_words = []
                # for line in file:
                #     parts = line.strip().split()
                #     if len(parts) >= 4:
                #         start_time = float(parts[2])
                #         word = parts[4]
                #         ctm_words.append((start_time, word))

            # Assuming local_audio_filename is something like '3dda4563-7850-286674-0001.wav'
            #ctm_filename = local_audio_filename  # Replace '.wav' with '.ctm'

            # Construct the path to the CTM file
            #ctm_file_path = f"{nfa_output_path}ctm/words/{ctm_filename}"

            # Read NFA output from the CTM file
            #with open(ctm_file_path, 'r') as file:
            #    nfa_data = json.load(file)  # Or process the file contents as needed

            task['data']['alignment'] = ctm_words

            # Format the result as expected by Label Studio
            task['result'] = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'textarea',
                'value': {
                    'text': [transcription]
                }
            }]
            task['score'] = 1.0  # If you have a scoring mechanism
                    # Capture the transcription result
            #
            task_id = task['id']
            updated_data = {
                'data': {
                    **task['data'],  # Keep existing data
                    'pred_text': transcription  # Update only pred_text
                }
            }

            response = update_task(task_id, updated_data, label_studio_url, api_key)
            print(response)
            #
            print(response)
            #task['data']['pred_text'] = transcription  # Update the pred_text field

        return tasks

    # def predict(self, tasks, **kwargs):
    #     for task in tasks:
    #         relative_audio_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
    #         full_audio_url = f"http://host.docker.internal:8080{relative_audio_url}"
    #         local_audio_filename = os.path.basename(relative_audio_url)
    #         local_audio_path = os.path.join('/tmp', local_audio_filename)

    #         # Downloading the audio file
    #         try:
    #             subprocess.run(['wget', '-O', local_audio_path, full_audio_url], check=True)
    #         except subprocess.CalledProcessError as e:
    #             logger.error(f"Error downloading audio file: {e}")
    #             continue

    #         # Transcribe audio file
    #         transcription = self.model.transcribe(paths2audio_files=[local_audio_path])[0]

    #         # Update task with transcription
    #         task['data']['pred_text'] = transcription

    #     return tasks

    def _bind_to_textarea(self):
        from_name, to_name, value = None, None, None
        for tag_name, tag_info in self.parsed_label_config.items():
            if tag_info['type'] == 'TextArea':
                from_name = tag_name
                to_name = tag_info['to_name'][0]
                value = tag_info['inputs'][0]['value']
                break
        if from_name is None:
            raise ValueError('ASR model expects <TextArea> tag to be presented in the label config.')
        return from_name, to_name, value



# import os
# import logging
# import nemo
# import nemo.collections.asr as nemo_asr
# import subprocess
# from label_studio_ml.model import LabelStudioMLBase
# from label_studio_ml.utils import DATA_UNDEFINED_NAME
# import nemo.collections.asr as nemo_asr
# #asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large")

# logger = logging.getLogger(__name__)
# #model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("stt_en_conformer_transducer_large")

# class NemoASR(LabelStudioMLBase):

#     #def __init__(self, model_name='stt_en_conformer_transducer_large', **kwargs):
#         #super(NemoASR, self).__init__(**kwargs)

#         # Find TextArea control tag and bind ASR model to it
#         #self.from_name, self.to_name, self.value = self._bind_to_textarea()

#         # This line will download pre-trained QuartzNet15x5 model from NVIDIA's NGC cloud and instantiate it for you
#         #logger.error("VERSION IS", nemo.__version__)
#         #self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
#     def __init__(self, model_name='nvidia/stt_en_fastconformer_transducer_large', **kwargs):
#         super(NemoASR, self).__init__(**kwargs)

#         # Find TextArea control tag and bind ASR model to it
#         self.from_name, self.to_name, self.value = self._bind_to_textarea()

#         # This line will download pre-trained QuartzNet15x5 model from NVIDIA's NGC cloud and instantiate it for you
#         #self.model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)
#         self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name)

# #    def predict(self, tasks, **kwargs):
# #        output = []
# #        audio_paths = []
# #        for task in tasks:
# #            audio_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
# #            audio_path = self.get_local_path(audio_url)
# #            #audio_path = "http://host.docker.internal:8080"+audio_url
# #            audio_paths.append(audio_path)
# #
# #        # run ASR
# #        transcriptions = self.model.transcribe(paths2audio_files=audio_paths)

# #        for transcription in transcriptions:
# #            output.append({
# #                'result': [{
# #                    'from_name': self.from_name,
# #                    'to_name': self.to_name,
# #                    'type': 'textarea',
# #                    'value': {
# #                        'text': [transcription]
# #                    }
# #                }],
# #                'score': 1.0
# #            })
# #        return output
#     def predict(self, tasks, **kwargs):
#         output = []
#         audio_paths = []
#         for task in tasks:
#             relative_audio_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
#             # Construct the full URL with host.docker.internal
#             full_audio_url = f"http://host.docker.internal:8080{relative_audio_url}"

#             # Define local path for the downloaded file
#             local_audio_filename = os.path.basename(relative_audio_url)
#             local_audio_path = os.path.join('/tmp', local_audio_filename)

#             # Download the file using wget
#             try:
#                 subprocess.run(['wget', '-O', local_audio_path, full_audio_url], check=True)
#             except subprocess.CalledProcessError as e:
#                 logger.error(f"Error downloading audio file: {e}")
#                 continue  # Skip this file if download fails

#             audio_paths.append(local_audio_path)
#         # run ASR
#         transcriptions = self.model.transcribe(paths2audio_files=audio_paths)
#         wer_implementation=1.0
#         for task, transcription in zip(tasks, transcriptions):
#             # Update 'pred_text' field in task data
#             task['data']['pred_text'] = transcription

#         return tasks

# #        for transcription in transcriptions:
# #            output.append(
# #                 {
# #                'text_first': [transcription],
# #                'result': [{
# #                    'from_name': self.from_name,
# #                    'to_name': self.to_name,
# #                    'type': 'textarea',
# #                    'value': {
# #                        'text': [transcription]
# #                 }
# #                        }
# #                           ],
# #                'score': wer_implementation
# #                        }
# #            )
# #        return output

#     def _bind_to_textarea(self):
#         from_name, to_name, value = None, None, None
#         for tag_name, tag_info in self.parsed_label_config.items():
#             if tag_info['type'] == 'TextArea':
#                 from_name = tag_name
#                 if len(tag_info['inputs']) > 1:
#                     logger.warning(
#                         'ASR model works with single Audio or AudioPlus input, '
#                         'but {0} found: {1}. We\'ll use only the first one'.format(
#                             len(tag_info['inputs']), ', '.join(tag_info['to_name'])))
#                 if tag_info['inputs'][0]['type'] not in ('Audio', 'AudioPlus'):
#                     raise ValueError('{0} tag expected to be of type Audio or AudioPlus, but type {1} found'.format(
#                         tag_info['to_name'][0], tag_info['inputs'][0]['type']))
#                 to_name = tag_info['to_name'][0]
#                 value = tag_info['inputs'][0]['value']
#         if from_name is None:
#             raise ValueError('ASR model expects <TextArea> tag to be presented in a label config.')
#         return from_name, to_name, value
