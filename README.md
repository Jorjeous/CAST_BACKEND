## What is the Label Studio ML backend?

The Label Studio ML backend is an SDK that lets you wrap your machine learning code and turn it into a web server.
The web server can be then connected to Label Studio to automate labeling tasks and dynamically retrieve pre-annotations from your model.

There are several use-cases for the ML backend:

- Pre-annotate data with a model
- Use active learning to select the most relevant data for labeling
- Interactive (AI-assisted) labeling
- Model fine-tuning based on recently annotated data

If you just need to load static pre-annotated data into Label Studio, running an ML backend might be overkill for you. Instead, you can [import preannotated data](https://labelstud.io/guide/predictions.html).


## Quickstart

Follow this example tutorial to create a ML backend service:

1. Install the latest Label Studio ML SDK:
   ```bash
   git clone https://gitlab-master.nvidia.com/gnemo/cast_backend.git
   cd label-studio-ml-backend/
   pip install -e .
   ```
   
2. Go to the NeMo directory:
    
   ```bash
   cd label_studio_ml/examples/nemo
   ```

   The directory structure should look like this:
   ```
    nemo/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── asr.py
    ├── _wsgi.py
    ├── requirements.txt
    ├── stt_en_fastconformer_hybrid_large_pc.nemo (LFS)
    ├── uwsgi.ini
    ├── align.py
    └── utils

    ```
    `Dockefile` and `docker-compose.yml` are used to run the ML backend with Docker.
    `asr.py` is the main file where you can implement your own training and inference logic.
    `_wsgi.py` is a helper file that is used to run the ML backend with Docker (you don't need to modify it)
    `requirements.txt` is a file with Python dependencies.
    `stt_en_fastconformer_hybrid_large_pc.nemo` is a english fasrconformer model, to start the server a bit faster. 

3. Do not forget to download LFS files. 
   ```bash
   git-lfs-fetch
   ```
4. Before building the docker image
 - run the Label studio application (refer to [CAST](https://gitlab-master.nvidia.com/gnemo/cast)) 
 - Log in or create new account 
 - Go to account and settings (http://localhost:8080/user/account)
 - Copy Access Token and past it in asr.py (there is variable API_TOKEN = 'your_token' )
 - If you running the Label Studio app on other port than 8080 - change it as well: label_studio_url = 'http://host.docker.internal:PORT'
 
4. When done - build the docker image. 
   ```bash
   docker compose build
   ```
5. Run the server:
   ```bash
   docker compose up
   ```
    The ML backend server will be available at `http://localhost:9090`. You can use this URL to connect it to Label Studio:
    Go to the project Settings > Machine Learning and Add a new ML backend.
    Enter your chosen name for ML model and paste the address as previously mentioned. Then, click on "Validate and Save."

    *Note: If you encounter an error message, it typically indicates that the server requires more time to load. In this case, click "Validate and Save" a few more times.
   
This particullar example works with Long Audio template.

*Note: you can start both server and label studio on remote machine. When done just create a tunnel from label studio application. Make shure it's not running on 9090. That port is binded for ML backend.

For more information about modification of your ML backend refer to: [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples)



## Implement prediction logic
In your model directory, locate the `asr.py` file.

The `model.py` file contains a class declaration inherited from `LabelStudioMLBase`. This class provides wrappers for the API methods that are used by Label Studio to communicate with the ML backend. You can override the methods to implement your own logic:
```python
def predict(self, tasks, context, **kwargs):
    """Make predictions for the tasks."""
    return predictions
```
The `predict` method is used to make predictions for the tasks. It uses the following:
- `tasks`: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
- `context`: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend) - for interactive labeling scenario
- `predictions`: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)

Once you implement the `predict` method, you can see predictions from the connected ML backend in Label Studio.

## Implement training logic
You can also implement the `fit` method to train your model. The `fit` method is typically used to train the model on the labeled data, although it can be used for any arbitrary operations that require data persistence (for example, storing labeled data in database, saving model weights, keeping LLM prompts history, etc).
By default, the `fit` method is called at any data action in Label Studio, like creating a new task or updating annotations. You can modify this behavior in Label Studio > Settings > Webhooks.

To implement the `fit` method, you need to override the `fit` method in your `model.py` file:
```python
def fit(self, event, data, **kwargs):
    """Train the model on the labeled data."""
    old_model = self.get('old_model')
    # write your logic to update the model
    self.set('new_model', new_model)
```
with 
- `event`: event type can be `'ANNOTATION_CREATED'`, `'ANNOTATION_UPDATED', etc.
- `data` the payload received from the event (check more on [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))

Additionally, there are two helper methods that you can use to store and retrieve data from the ML backend:
- `self.set(key, value)` - store data in the ML backend
- `self.get(key)` - retrieve data from the ML backend

Both methods can be used elsewhere in the ML backend code, for example, in the `predict` method to get the new model weights.

## Other methods and parameters
Other methods and parameters are available within the `LabelStudioMLBase` class:

- `self.label_config` - returns the [Label Studio labeling config](https://labelstud.io/guide/setup.html) as XML string.
- `self.parsed_label_config` - returns the [Label Studio labeling config](https://labelstud.io/guide/setup.html) as JSON.
- `self.model_version` - returns the current model version.


## Run without Docker

To run without docker (for example, for debugging purposes), you can use the following command:
```bash
pip install -r my_ml_backend
label-studio-ml start my_ml_backend
```

### Modify the port
To modify the port, use the `-p` parameter:
```bash
label-studio-ml start my_ml_backend -p 9091
```

## Deploy your ML backend to GCP

Before you start:
1. Install [gcloud](https://cloud.google.com/sdk/docs/install)
2. Init billing for account if it's not [activated](https://console.cloud.google.com/project/_/billing/enable)
3. Init gcloud, type the following commands and login in browser:
```bash
gcloud auth login
```
4. Activate your Cloud Build API
5. Find your GCP project ID
6. (Optional) Add GCP_REGION with your default region to your ENV variables 

To start deployment:
1. Create your own ML backend
2. Start deployment to GCP:
```bash
label-studio-ml deploy gcp {ml-backend-local-dir} \
--from={model-python-script} \
--gcp-project-id {gcp-project-id} \
--label-studio-host {https://app.heartex.com} \
--label-studio-api-key {YOUR-LABEL-STUDIO-API-KEY}
```
3. After label studio deploys the model - you will get model endpoint in console.
