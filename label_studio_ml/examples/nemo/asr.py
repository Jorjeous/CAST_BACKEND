import os
import logging
import subprocess
import nemo.collections.asr as nemo_asr
import json
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME
import requests

# Configuration
API_TOKEN = '6f1303c16baad634fb7eb4291270c36249ab6fa9'
label_studio_url = 'http://host.docker.internal:8080'

# Setup logging
#logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Cache for storing task results
task_results_cache = {}

def update_task(task_id, updated_data):
    logger.info("Updating task in Label Studio.")
    update_url = f"{label_studio_url}/api/tasks/{task_id}"
    headers = {'Authorization': f'Token {API_TOKEN}'}
    response = requests.patch(update_url, json=updated_data, headers=headers)
    if response.status_code == 200:
        logger.info(f"Task {task_id} updated successfully.")
    else:
        logger.error(f"Failed to update task {task_id}. Status code: {response.status_code}")
        return response.json()

class NemoASR(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(NemoASR, self).__init__(**kwargs)
        self.processed_audio_path = {}
        self.from_name, self.to_name, self.value = self._bind_to_textarea()
        self.model_name_base = 'nvidia/stt_en_fastconformer_hybrid_large_pc'
        # Initialize NeMo ASR model within __init__
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(self.model_name_base)
        logger.info("NemoASR model initialized.")

    def process_ctm_file(self, ctm_file_path):
        annotations = []
        current_annotation = {"start": None, "text": "", "author": "author"}
        segment_duration_min = 10.0  # Min target duration in seconds for each segment
        segment_duration_max = 20.0  # Max duration in seconds for each segment

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

                    if current_annotation["start"] is not None:
                        current_duration = start_time - current_annotation["start"]

                    if current_annotation["text"] and (is_end_of_sentence or current_duration >= segment_duration_max):
                        if current_duration >= segment_duration_min or is_end_of_sentence:
                            current_annotation["text"] += f"{word} "
                            current_annotation["end"] = end_time
                            annotations.append(current_annotation)
                            current_annotation = {"start": None, "text": "", "author": self.model_name_base, "end": None}
                            continue
                        else:
                            current_annotation["end"] = previous_end_time

                    if current_annotation["start"] is None:
                        current_annotation["start"] = start_time

                    current_annotation["text"] += f"{word} "
                    previous_end_time = end_time  # Update previous_end_time after adding the word

            # Ensure the last annotation is added if it contains text and hasn't been added yet
            if current_annotation["text"]:
                if "end" not in current_annotation or current_annotation["end"] is None:
                    current_annotation["end"] = previous_end_time
                annotations.append(current_annotation)

        return annotations

    def predict(self, tasks, **kwargs):
        for task in tasks:
            task_id = task['id']
            logger.info(f"Processing task {task_id}.")

            relative_audio_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
            local_audio_filename = os.path.basename(relative_audio_url)
            local_audio_path = os.path.join('/tmp', local_audio_filename)
            full_audio_url = f"http://host.docker.internal:8080{relative_audio_url}"
            if local_audio_path not in self.processed_audio_path:
                logger.info(f"Downloading audio file for task {task_id}.")
                try:
                    subprocess.run(['wget', '--header', f'Authorization: Token {API_TOKEN}', '-O', local_audio_path, full_audio_url], check=True)
                    logger.info(f"Audio file for task {task_id} downloaded successfully.")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error downloading audio file for task {task_id}: {e}")
                    continue
                self.processed_audio_path[local_audio_path] = True
            
            # Download audio file with authentication

            transcription = self.model.transcribe(paths2audio_files=[local_audio_path])[0][0]
            logger.info(f"Transcripted, continueing to NFA processing.")
            # Process CTM file and update task result
            nfa_output_path = "/tmp/nfaout/"
            manifest_filepath = os.path.join('/tmp', 'manifest.json')
            with open(manifest_filepath, 'w') as f:
                json.dump({"audio_filepath": local_audio_path, "text": transcription}, f)
            
            subprocess.run([f"/app/align.py", 
                            f"pretrained_name=stt_en_fastconformer_hybrid_large_pc", 
                            f"manifest_filepath={manifest_filepath}", 
                            f"output_dir={nfa_output_path}", 
                             "additional_segment_grouping_separator=|"
                             ], check=True)
            logger.info(f"NFA Done.")
            npath = os.path.join(nfa_output_path, 'ctm/words/', local_audio_filename.replace('.wav', '.ctm'))
            ctm_words = self.process_ctm_file(npath)

            task['data']['alignment'] = ctm_words
            task['result'] = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'textarea',
                'value': {'text': [transcription]}
            }]
            task['score'] = 1.0

            # Cache the result
            task_results_cache[task_id] = task['result']

            # Optionally update task in Label Studio
            updated_data = {'data': {**task['data'], 'pred_text': transcription}}
            update_task(task_id, updated_data)

        return tasks

    def _bind_to_textarea(self):
        for tag_name, tag_info in self.parsed_label_config.items():
            if tag_info['type'] == 'TextArea':
                return tag_name, tag_info['to_name'][0], tag_info['inputs'][0]['value']
        raise ValueError('ASR model expects <TextArea> tag to be presented in the label config.')

