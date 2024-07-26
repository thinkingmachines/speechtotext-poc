# Troubleshooting
## Debugging
### Logs Location

- Development environment logs:

  - Cloud Run logs: GCP Console > Cloud Run > set-speechtotext-dev > Logs
  - Application logs: `gs://set-speechtotext-logs-dev/application-logs/`

- Production environment logs:

  - Cloud Run logs: GCP Console > Cloud Run > set-speechtotext-prod > Logs
  - Application logs: `gs://set-speechtotext-logs-prod/application-logs/`

To view logs locally, use the following gcloud command:
```zsh
gcloud run logs read --service=set-speechtotext-dev --project=your-project-id
```

## Common Issues

1. Audio file format incompatibility

   - Symptom: Preprocessing fails for certain audio files
   - Solution: Ensure all input audio files are in WAV format with a sample rate of 16kHz.

2. Out of memory errors during model training

   - Symptom: Training script crashes with OOM error
   - Solution: Reduce batch size or use a machine with more memory. Consider using gradient accumulation for larger models.

3. Low transcription accuracy on certain accents

   - Symptom: Model performs poorly on specific regional accents
   - Solution: Fine-tune the model on a dataset that includes more diverse accents.

4. Slow inference times in production

   - Symptom: API responses are taking too long
   - Solution:

     1. Check the Cloud Run instance configuration and increase resources if necessary.
     2. Optimize for faster processing.
     3. Consider implementing batch processing for large volumes of requests.

5. Inconsistent results between local and cloud environments

   - Symptom: Model behaves differently when deployed vs. local testing
   - Solution: Ensure all environment variables are correctly set in both environments.

6. Data preprocessing pipeline failures

   - Symptom: Preprocessing jobs fail or produce inconsistent results
   - Solution:

     1. Check the input data quality and format.
     2. Verify that all required libraries are correctly installed and up-to-date.
     3. Review and update the preprocessing scripts in src/preprocessing/ if necessary.

7. API authentication issues

   - Symptom: Unauthorized access errors when trying to use the speech-to-text API
   - Solution:

     1. Verify that the correct API keys or tokens are being used.
     2. Check the expiration dates of the authentication credentials.
     3. Ensure that the service account has the necessary permissions in GCP.

If you encounter persistent issues not covered here, please consult the project's internal documentation on Confluence or reach out to the team for support.
