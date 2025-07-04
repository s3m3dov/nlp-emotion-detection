import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import logging
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the emotions
EMOTIONS = ['anger', 'fear', 'joy', 'sadness', 'surprise']

class EmotionDetector:
    def __init__(self, model_path='./results', model_name=None, fallback_path=None):
        """
        Initialize the emotion detector with a pre-trained model

        Args:
            model_path (str): Path to the saved model directory
            model_name (str, optional): Base model name to use if no fine-tuned model is found
            fallback_path (str, optional): Fallback path if main path fails
        """
        self.model_name = model_name or 'bert-base-uncased'
        self.tokenizer = None
        self.model = None

        # Try loading from the specified path
        if self._try_load_model(model_path):
            return

        # Try fallback path if provided
        if fallback_path and self._try_load_model(fallback_path):
            return

        # Final fallback to base model
        logger.info(f"Using base {self.model_name} model. Note: This model is not trained for emotion detection and will give random results.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(EMOTIONS),
            problem_type="multi_label_classification"
        )
        self.model.eval()

    def _try_load_model(self, path):
        """Try to load model from the specified path"""
        try:
            # Check if directory exists
            if not os.path.exists(path):
                logger.warning(f"Model path '{path}' does not exist")
                return False

            # Check if model files exist
            model_files = os.listdir(path)
            required_files = ['config.json']
            model_file_exists = any(f in model_files for f in ['pytorch_model.bin', 'model.safetensors'])

            if not all(f in model_files for f in required_files) or not model_file_exists:
                logger.warning(f"Missing required model files in '{path}'")
                return False

            logger.info(f"Loading model from {path}")
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path,
                num_labels=len(EMOTIONS),
                problem_type="multi_label_classification"
            )
            self.model.eval()
            return True

        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}")
            return False

    def predict_text(self, text, threshold=0.5):
        """
        Predict emotions in the given text

        Args:
            text (str): The input text to analyze
            threshold (float): Probability threshold for emotion detection

        Returns:
            dict: Dictionary with detected emotions and their probabilities
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model has been loaded successfully")

        # Tokenize the text
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits).numpy()[0]

        # Create results dictionary
        result = {
            'text': text,
            'emotions': {},
            'detected_emotions': [],
            'probabilities': probabilities
        }

        # Add probabilities for each emotion and track detected emotions with their probabilities
        detected_emotions_with_probs = []
        for i, emotion in enumerate(EMOTIONS):
            prob = float(probabilities[i])
            result['emotions'][emotion] = prob
            if prob >= threshold:
                detected_emotions_with_probs.append((emotion, prob))

        # Sort detected emotions by probability (highest first)
        detected_emotions_with_probs.sort(key=lambda x: x[1], reverse=True)
        result['detected_emotions'] = [emotion for emotion, _ in detected_emotions_with_probs]

        return result

    def predict_batch(self, texts, threshold=0.5):
        """
        Predict emotions for a batch of texts

        Args:
            texts (list): List of strings to analyze
            threshold (float): Probability threshold for emotion detection

        Returns:
            list: List of prediction results
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model has been loaded successfully")

        results = []

        # Process in batches to avoid memory issues
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize the batch
            inputs = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Make predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(logits).numpy()

            # Process each prediction in the batch
            for j, text in enumerate(batch_texts):
                probs = probabilities[j]

                # Create results dictionary for each text
                result = {
                    'text': text,
                    'emotions': {},
                    'detected_emotions': [],
                    'probabilities': probs
                }

                # Add probabilities for each emotion and track detected emotions with their probabilities
                detected_emotions_with_probs = []
                for k, emotion in enumerate(EMOTIONS):
                    prob = float(probs[k])
                    result['emotions'][emotion] = prob
                    if prob >= threshold:
                        detected_emotions_with_probs.append((emotion, prob))

                # Sort detected emotions by probability (highest first)
                detected_emotions_with_probs.sort(key=lambda x: x[1], reverse=True)
                result['detected_emotions'] = [emotion for emotion, _ in detected_emotions_with_probs]

                results.append(result)

        return results

    def get_human_readable_predictions(self, csv_file_path, threshold=0.5):
        """
        Generate human-readable emotion predictions for texts in a CSV file

        Args:
            csv_file_path (str): Path to the CSV file containing texts to analyze
            threshold (float): Probability threshold for emotion detection

        Returns:
            pd.DataFrame: DataFrame with texts and their predicted emotions
        """
        try:
            # Load the CSV file
            logger.info(f"Loading CSV file from {csv_file_path}")
            df = pd.read_csv(csv_file_path)

            if 'text' not in df.columns:
                raise ValueError("CSV file must contain a 'text' column")

            texts = df['text'].tolist()
            logger.info(f"Loaded {len(texts)} texts from CSV")

            # Make predictions
            logger.info("Making predictions...")
            results = self.predict_batch(texts, threshold)

            # Create a new DataFrame with prediction results
            output_data = []
            for result in results:
                row = {
                    'text': result['text'],
                    'detected_emotions': ', '.join(result['detected_emotions']) if result['detected_emotions'] else 'None'
                }

                # Add individual emotion probabilities
                for emotion, prob in result['emotions'].items():
                    row[f'{emotion}_probability'] = f"{prob:.4f}"

                output_data.append(row)

            output_df = pd.DataFrame(output_data)
            logger.info("Human-readable predictions generated")
            return output_df

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

def predict(csv_file_path, model_path='./results', model_name=None, threshold=0.5):
    """
    Predict emotions for texts in a CSV file

    Args:
        csv_file_path (str): Path to the CSV file containing texts to analyze
        model_path (str): Path to the model directory
        model_name (str, optional): Base model name to use if no fine-tuned model is found
        threshold (float): Probability threshold for emotion detection

    Returns:
        numpy.ndarray: Array of binary prediction arrays for each item in the CSV
                      Each prediction is an array of 0s and 1s for
                      [anger, fear, joy, sadness, surprise], where 1 indicates
                      the emotion is detected (probability >= threshold)
    """
    try:
        # Load the CSV file
        logger.info(f"Loading CSV file from {csv_file_path}")
        df = pd.read_csv(csv_file_path)

        if 'text' not in df.columns:
            raise ValueError("CSV file must contain a 'text' column")

        texts = df['text'].tolist()
        logger.info(f"Loaded {len(texts)} texts from CSV")

        # Initialize the emotion detector
        detector = EmotionDetector(model_path=model_path, model_name=model_name)

        # Make predictions
        logger.info("Making predictions...")
        results = detector.predict_batch(texts, threshold)

        # Extract probabilities for each emotion
        probabilities = np.array([result['probabilities'] for result in results])

        # Convert probabilities to binary values (0s and 1s)
        predictions = (probabilities >= threshold).astype(int)

        logger.info("Binary predictions complete")
        return predictions

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def predict_and_print(csv_file_path, model_path='./results', model_name=None, threshold=0.5):
    """
    Helper function to predict emotions and display results in human-readable format

    Args:
        csv_file_path (str): Path to the CSV file containing texts to analyze
        model_path (str): Path to the model directory
        model_name (str, optional): Base model name to use if no fine-tuned model is found
        threshold (float): Probability threshold for emotion detection

    Returns:
        pd.DataFrame: DataFrame with texts and their predicted emotions
    """
    # Initialize the emotion detector
    detector = EmotionDetector(model_path=model_path, model_name=model_name)

    # Get human-readable predictions
    results_df = detector.get_human_readable_predictions(csv_file_path, threshold)

    # Display the results
    print(f"\nEmotion Detection Results for {csv_file_path}:")
    for _, row in results_df.iterrows():
        print(f"\nText: {row['text']}")
        print(f"Detected emotions: {row['detected_emotions']}")
        print("Emotion probabilities:")
        for emotion in EMOTIONS:
            print(f"  {emotion}: {row[f'{emotion}_probability']}")

    return results_df

def main():
    """
    Main function to demonstrate emotion detection
    """
    # Initialize the emotion detector with multiple potential paths
    try:
        detector = EmotionDetector(
            model_path='./results',
            model_name='bert-base-uncased',
            fallback_path='./model'
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return

    # Example texts
    example_texts = [
        "I'm so happy to see you after all this time!",
        "The news of the accident made me very sad.",
        "I was terrified when I heard the loud noise at night.",
        "His rude comment made me really angry.",
        "I was completely surprised by the unexpected gift."
    ]

    # Test the detector with example texts
    for text in example_texts:
        try:
            result = detector.predict_text(text)
            print(f"\nText: {text}")
            print("Detected emotions:", ", ".join(result['detected_emotions']) if result['detected_emotions'] else "None")
            print("Emotion probabilities:")
            for emotion, prob in result['emotions'].items():
                print(f"  {emotion}: {prob:.4f}")
        except Exception as e:
            logger.error(f"Error predicting emotions for '{text}': {str(e)}")

    # Example of using the predict function with a CSV file
    print("\nTesting CSV prediction function:")
    try:
        # If you have a test CSV file, uncomment and modify this section:
        test_csv = "data/test.csv"
        predictions = predict(test_csv, model_path="./results", model_name="roberta-base")
        print(f"Predictions shape: {predictions.shape}")
        print("First 3 predictions:")
        for i in range(min(3, len(predictions))):
            print(f"  {EMOTIONS} = {predictions[i]}")
        print("To use the CSV prediction function, call predict('your_csv_file.csv')")
    except Exception as e:
        logger.error(f"Error testing CSV prediction: {str(e)}")

if __name__ == "__main__":
    # Example usage with specific model path and model name
    result = predict(
        "data/test.csv",
        model_path="./results/roberta-base_vanilla_out",
        model_name="roberta-base"
    )
    print(result)

    # Uncomment to get human-readable results
    # predict_and_print(
    #     "data/test.csv",
    #     model_path="./results/roberta-base_vanilla_out",
    #     model_name="roberta-base"
    # )