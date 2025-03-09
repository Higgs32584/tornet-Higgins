
from huggingface_hub import hf_hub_download
model_file = hf_hub_download(repo_id="tornet-ml/tornado_detector_baseline_v1", 
                             filename="tornado_detector_baseline.keras")

# Alternatively, you can manually download the .keras file and put in the ../models/ directory
# https://huggingface.co/tornet-ml/tornado_detector_baseline_v1
#model_file = '../models/tornado_detector_baseline.keras' 

# Load pretrained model
#cnn = keras.models.load_model(model_file,compile=False)
