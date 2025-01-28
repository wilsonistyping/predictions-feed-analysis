# Installation Instructions

1. Create a virtual environment in the project directory.
`python -m venv .env`

2. Activate the virtual environment.
Linux/MacOS: `source .env/bin/activate`
Windows: `.env\Scripts\activate`

3. Install Tranformers and Pytorch
`pip install 'transformers[torch]'`

4. Test if installation was successful by running the following command:
`python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"`

(This should download a pretrained model and output a sentiment analysis result, like this:
[{'label': 'POSITIVE', 'score': 0.9998704791069031}])

---
Resources:
https://www.youtube.com/watch?v=QEaBAZQCtwE

Other things to install:
pip install scikit-learn transformers torch pandas numpy
