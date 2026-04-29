Method 1: Rule-Based Quiz Generation

This module implements a deterministic NLP-based approach to generate multiple choice questions (MCQs) directly from input study material. The pipeline processes raw text using preprocessing, sentence tokenization, topic modeling (LDA / TF-IDF fallback), and keyword extraction techniques.

Key steps include:

Cleaning and filtering input text
Extracting meaningful sentences
Identifying important keywords using TF-IDF and POS tagging
Generating fill-in-the-blank questions using templates
Creating distractor options from keyword pools

This approach ensures that all generated questions are strictly derived from the provided content, making it reliable and free from hallucination. However, the variation and linguistic quality of questions are limited compared to LLM-based methods.
