# Multi-Label Emotion Detection Using NLP

This repository demonstrates the process of training and evaluating models to detect multiple emotions in text.

## Overview

1. Data loading and exploration
2. Emotion distribution analysis
3. Text preprocessing
4. Traditional ML approach with Naive Bayes
5. Deep learning approach with BERT
6. Model evaluation and saving
7. Loading pretrained models and testing on test data

## Bookmarks

- [BERT best performance](https://github.com/s3m3dov/nlp-emotion-detection/blob/cfcb895a81be2478469ae08c8072a8fe6638c8c1/multi-label-emotion-detection.ipynb)

---

## Task Description (Track A)

This project is part of the 2025 SemEval Shared Task 11 (Muhammad et al. 2025) for the course "Introduction to Natural Language Processing" (Prof. Dr. Daniel Braun, Summer 2025).

**Track A (Multi-label Emotion Detection):**  
Given a target text snippet, predict the perceived emotion(s) of the speaker. Specifically, select whether each of the following emotions apply: joy, sadness, fear, anger, surprise, or disgust. In other words, label the text snippet with:  
- joy (1) or no joy (0)  
- sadness (1) or no sadness (0)  
- anger (1) or no anger (0)  
- surprise (1) or no surprise (0)  
- disgust (1) or no disgust (0)

### Project Milestones

1. **Team formation and track selection** (not graded)
2. **Project proposal** (20 pts): Task description, motivation, dataset, methodology, evaluation plan.
3. **Technical solution** (50 pts): Implement a Python script for the classification task. The main.py must provide a `predict` function that takes a CSV file and returns an array of predictions.
4. **Project report** (30 pts): Introduction, technical solution, evaluation, reflection.

### Dataset

Based on the 2025 SemEval Shared Task 11 data. Training datasets are provided for both tracks.

### Evaluation

Evaluation is based on accuracy, technical quality, documentation, and difficulty of the solution. See the official worksheet for detailed rubrics.

### References

Muhammad, Shamsuddeen Hassan et al. (2025). “SemEval Task 11: Bridging the Gap in Text-Based Emotion Detection”. In: Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025). Vienna, Austria: Association for Computational Linguistics.

---

