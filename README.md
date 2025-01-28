# Installation Instructions

1. Create a virtual environment in the project directory.
`python -m venv .env`

2. Activate the virtual environment.
Linux/MacOS: `source .env/bin/activate`
Windows: `.env/Scripts/activate`

3. Install Libraries
`pip install transformers torch pandas numpy`

4. Test if installation was successful by running the following command:
`python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('Yay this works'))"`

(This should download a pretrained model and output a sentiment analysis result, like this:
[{'label': 'POSITIVE', 'score': 0.9998704791069031}])

--- OPTIONAL ---
For the models to use GPU power instead of CPU power, we need CUDA installed.
(THIS ONLY WORKS FOR WINDOWS, not MacOS.)
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

Additionally, it needs a PyTorch install that uses the CUDA version we have (install command found/customizable here):
https://pytorch.org/get-started/locally/

To check if CUDA is installed:
`nvcc --version`

---
Resources:
https://www.youtube.com/watch?v=QEaBAZQCtwE
https://huggingface.co/docs/transformers/installation
---
