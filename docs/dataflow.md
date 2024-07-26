# Data Flow Documentation
The set-speechtotext-poc project follows a data pipeline that processes audio files and converts them into text. Below is a description of the end-to-end data flow:

1. Data Ingestion: Raw audio files are uploaded to a GCS bucket (raw-audio-bucket).
2. Preprocessing: The preprocess_audio Cloud Function is triggered by new uploads, which normalizes and segments the audio.
3. Feature Extraction: The extract_features Dataflow job processes the normalized audio files and extracts MFCC features.
4. Model Inference: The speech_to_text_model Cloud Run service takes the extracted features and generates text transcriptions.
5. Post-processing: The post_process_text Cloud Function cleans and formats the transcribed text.
Storage: Final transcriptions are stored in a Cloud SQL database and also in a GCS bucket for backup.

## Critical components:

1. Preprocessing: `preprocess_audio`.
2. Model Inference: `speech_to_text_model`.
3. Post-processing: `post_process_text`.
4. Storage: `transcriptions` Cloud SQL database, `processed-transcriptions` GCS bucket.


## Input and Output Data
### Input Data
#### Input A: Raw Audio Files

- Description: Unprocessed audio recordings in various formats (WAV, MP3, etc.)
- File Location: Cloud Service Database or on your Local Machine.

#### Input B: Pre-trained Model Weights

- Description: Initial weights for the speech recognition model
- File Location: Stored in Transformen Models

### Output Data
#### Output A: Transcribed Text

- Description: Text transcriptions of the input audio files
- File Location: Cloud SQL database transcriptions table

#### Output B: Model Performance Metrics

- Description: Accuracy, WER, and other relevant metrics for model evaluation
- File Location: Cloud Service Database or on your Local Machine.

This data flow ensures that raw audio inputs are processed, transcribed, and stored efficiently, with the final output being accessible through both database queries and API endpoints. The system is designed to scale with increasing data volumes and provides flexibility for future improvements and model updates.
