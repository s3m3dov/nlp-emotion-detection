import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the emotions
EMOTIONS = ['anger', 'fear', 'joy', 'sadness', 'surprise']

class EmotionDetector:
    def __init__(self, model_path='./results', fallback_path=None):
        """
        Initialize the emotion detector with a pre-trained model

        Args:
            model_path (str): Path to the saved model directory
            fallback_path (str, optional): Fallback path if main path fails
        """
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None

        # Try loading from the specified path
        if self._try_load_model(model_path):
            return

        # Try fallback path if provided
        if fallback_path and self._try_load_model(fallback_path):
            return

        # Final fallback to base model
        logger.info("Using base BERT model. Note: This model is not trained for emotion detection and will give random results.")
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
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
            self.model = BertForSequenceClassification.from_pretrained(
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
        if self.model is None:
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

        # Add probabilities for each emotion
        for i, emotion in enumerate(EMOTIONS):
            prob = float(probabilities[i])
            result['emotions'][emotion] = prob
            if prob >= threshold:
                result['detected_emotions'].append(emotion)

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
        if self.model is None:
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

                # Add probabilities for each emotion
                for k, emotion in enumerate(EMOTIONS):
                    prob = float(probs[k])
                    result['emotions'][emotion] = prob
                    if prob >= threshold:
                        result['detected_emotions'].append(emotion)

                results.append(result)

        return results

def save_model_from_trainer(trainer, output_dir='./results'):
    """
    Save the model from a Hugging Face trainer object

    This function should be called from the notebook after training

    Args:
        trainer: Hugging Face Trainer object with trained model
        output_dir (str): Directory to save the model
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the model
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")

def predict(csv_file_path, model_path='./results', threshold=0.5):
    """
    Predict emotions for texts in a CSV file

    Args:
        csv_file_path (str): Path to the CSV file containing texts to analyze
        model_path (str): Path to the model directory
        threshold (float): Probability threshold for emotion detection

    Returns:
        numpy.ndarray: Array of prediction arrays for each item in the CSV
                      Each prediction is an array of probabilities for
                      [anger, fear, joy, sadness, surprise]
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
        detector = EmotionDetector(model_path=model_path)

        # Make predictions
        logger.info("Making predictions...")
        results = detector.predict_batch(texts, threshold)

        # Extract probabilities for each emotion
        predictions = np.array([result['probabilities'] for result in results])

        logger.info("Predictions complete")
        return predictions

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def main():
    """
    Main function to demonstrate emotion detection
    """
    # Initialize the emotion detector with multiple potential paths
    try:
        detector = EmotionDetector(
            model_path='./results',
            fallback_path='./model'  # Try an alternative path if exists
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
        predictions = predict(test_csv)
        print(f"Predictions shape: {predictions.shape}")
        print("First 3 predictions:")
        for i in range(min(3, len(predictions))):
            print(f"  {EMOTIONS} = {predictions[i]}")
        print("To use the CSV prediction function, call predict('your_csv_file.csv')")
    except Exception as e:
        logger.error(f"Error testing CSV prediction: {str(e)}")

if __name__ == "__main__":
    result = predict("data/test.csv")
    print(result)
