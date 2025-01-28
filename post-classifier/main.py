import csv
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Initialize the sentiment and zero-shot classification pipelines
sentiment_classifier = pipeline("sentiment-analysis")
zero_shot_classifier = pipeline("zero-shot-classification")

def analyze_prediction_type(text):
    # Define candidate prediction types based on your CSV categories
    candidate_labels = [
        "betting market prediction",
        "polling-based prediction",
        "intuition-based prediction",
        "expert analysis",
        "crowd sentiment",
        "not a prediction"
    ]
    
    result = zero_shot_classifier(text, candidate_labels)
    return result['labels'][0], result['scores'][0]

def analyze_certainty(text):
    # Words indicating certainty levels
    certainty_indicators = {
        'high': ['definitely', 'certainly', 'absolutely', '100 percent', 'guarantee', 'landslide'],
        'medium': ['likely', 'probably', 'predict', 'might', 'could'],
        'low': ['maybe', 'possibly', 'uncertain', 'unclear']
    }
    
    text_lower = text.lower()
    
    # Check for certainty indicators
    for level, words in certainty_indicators.items():
        if any(word in text_lower for word in words):
            return level
    return 'unspecified'

def process_document(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    print("Available columns:", df.columns.tolist())
    
    # Initialize lists to store results
    prediction_types = []
    prediction_confidences = []
    certainty_levels = []
    sentiments = []
    
    # Process each text entry
    for text in df['text']:  # Changed from 'Text' to 'text'
        print(text)
        # Get prediction type and confidence
        pred_type, confidence = analyze_prediction_type(text)
        prediction_types.append(pred_type)
        prediction_confidences.append(confidence)
        
        # Get certainty level
        certainty = analyze_certainty(text)
        certainty_levels.append(certainty)
        
        # Get sentiment
        sentiment = sentiment_classifier(text)[0]
        sentiments.append(sentiment['label'])
    
    # Add new columns to DataFrame
    df['prediction_type'] = prediction_types
    df['prediction_confidence'] = prediction_confidences
    df['certainty_level'] = certainty_levels
    df['sentiment'] = sentiments
    
    # Select relevant columns for output
    output_df = df[['text', 'Link to post', 'prediction_type',
                    'prediction_confidence', 'certainty_level', 'sentiment']]
    
    # Export to JSON
    output_df.to_json("output/prediction_analysis.json", orient='records', indent=2)
    return output_df

# Process the document
if __name__ == "__main__":
    process_document("data/truth-social-logan.csv")
