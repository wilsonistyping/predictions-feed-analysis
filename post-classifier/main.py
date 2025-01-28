import pandas as pd
from transformers import pipeline
import numpy as np

# Initialize the sentiment and zero-shot classification pipelines
sentiment_classifier = pipeline("sentiment-analysis")
zero_shot_classifier = pipeline("zero-shot-classification")

def analyze_type(text):
    # Define candidate prediction types based on CSV categories
    type_labels = [
        "astrology",
        "tarot",
        "spiritual or religious",
        "betting market",
        "election poll",
        "election forecast",
        "gut feeling",
        "crowd sentiment",
        "historical event reasoning",
    ]
    type_result = zero_shot_classifier(text, type_labels)
    return type_result['labels'][0:2], type_result['scores'][0:2]

def analyze_prediction(text):
    prediction_labels = [
        "prediction",
        "not a prediction"
    ]

    prediction_result = zero_shot_classifier(text, prediction_labels)
    return prediction_result['labels'][0], prediction_result['scores'][0]

def analyze_certainty(text):
    certainty_labels = [
        "high certainty",
        "medium certainty",
        "low certainty"
    ]
    certainty_result = zero_shot_classifier(text, certainty_labels)
    return certainty_result['labels'][0], certainty_result['scores'][0]

def process_document(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    # Can comment this next line out, it's mostly for debugging the column keys
    print("Available columns:", df.columns.tolist())
    
    # Initialize lists to store results
    content_types = []
    content_types_confidences = []

    is_a_prediction = []
    is_a_prediction_confidences = []

    certainty_levels = []
    certainty_confidences = []
    sentiments = []
    
    # Process each text entry
    for text in df['Text of post']: 
        print(text)
        # If the text is empty or nan, skip it
        if pd.isna(text):
            content_types.append("no text")
            content_types_confidences.append(0)
            is_a_prediction.append("no text")
            is_a_prediction_confidences.append(0)
            certainty_levels.append("unspecified")
            certainty_confidences.append(0)
            sentiments.append("unspecified")
            continue
        # If the text exceeds the maximum token limit (should be 512 but going 255 to minimize chances), skip it
        elif len(text.split()) > 255:
            content_types.append("no text")
            content_types_confidences.append(0)
            is_a_prediction.append("no text")
            is_a_prediction_confidences.append(0)
            certainty_levels.append("unspecified")
            certainty_confidences.append(0)
            sentiments.append("unspecified")
            continue

        # Get content type
        content_type, content_confidence = analyze_type(text)
        content_types.append(content_type)
        content_types_confidences.append(content_confidence)

        # Get prediction confidence
        is_pred, pred_confidence = analyze_prediction(text)
        is_a_prediction.append(is_pred)
        is_a_prediction_confidences.append(pred_confidence)

        # Get expressed certainty level
        certainty, certainty_confidence = analyze_certainty(text)
        certainty_levels.append(certainty)
        certainty_confidences.append(certainty_confidence)
        
        # Get sentiment
        sentiment = sentiment_classifier(text)[0]
        sentiments.append(sentiment['label'])
    
    # Add new columns to DataFrame
    df['content_type'] = content_types
    df['content_type_confidence'] = content_types_confidences

    df['is_a_prediction'] = is_a_prediction
    df['prediction_confidence'] = is_a_prediction_confidences

    df['certainty_level'] = certainty_levels
    df['certainty_confidence'] = certainty_confidences
    
    df['sentiment'] = sentiments
    
    # Combine stuff into a nested structure
    df['content'] = df.apply(lambda row: [{'type': row['content_type'], 'confidence': row['content_type_confidence']}], axis=1)
    df['is_prediction'] = df.apply(lambda row: [{'is_a_prediction': row['is_a_prediction'], 'prediction_confidence': row['prediction_confidence']}], axis=1)
    df['certainty'] = df.apply(lambda row: [{'certainty_level': row['certainty_level'], 'certainty_confidence': row['certainty_confidence']}], axis=1)
    

    # Select relevant columns for output
    output_df = df[['Platform', 'Text of post', 'Link to post', 'content', 'is_prediction', 'certainty', 'sentiment']]
    
    # Export to JSON
    output_df.to_json("output/gali-prediction_analysis.json", orient='records', indent=2)
    return output_df

# Process the document
if __name__ == "__main__":
    process_document("data/gali-sample.csv")
