import librosa
import numpy as np
import soundfile as sf
import torch
from processor import tokenize_text
from transformers import Pipeline

TARGET_SAMPLING_RATE = 16_000


class RatchadaPipeline(Pipeline):
    def __init__(self, model, tokenizer, feature_extractor, device=None, **kwargs):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=device,
        )
        self.generate_kwargs = kwargs.get("generate_kwargs", {})

    def _sanitize_parameters(self, return_timestamps=None, **kwargs):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}

        if return_timestamps is not None:
            forward_params["return_timestamps"] = return_timestamps

        if kwargs:
            for key, value in kwargs.items():
                if key in self.generate_kwargs:
                    forward_params[key] = value
                else:
                    forward_params[key] = value

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs):
        if isinstance(inputs, str):
            audio, sr = sf.read(inputs)
            if sr != TARGET_SAMPLING_RATE:
                audio = librosa.resample(
                    audio, orig_sr=sr, target_sr=TARGET_SAMPLING_RATE
                )
        elif isinstance(inputs, np.ndarray):
            audio = inputs
        else:
            raise ValueError("Input must be a file path or a numpy array")

        inputs = self.feature_extractor(
            audio, sampling_rate=TARGET_SAMPLING_RATE, return_tensors="pt"
        ).to(self.device)
        return inputs

    def _forward(self, model_inputs, **forward_params):
        generate_kwargs = {**self.generate_kwargs, **forward_params}
        outputs = self.model.generate(**model_inputs, **generate_kwargs)
        return outputs

    def postprocess(self, model_outputs):
        transcription = self.tokenizer.batch_decode(
            model_outputs, skip_special_tokens=True
        )[0]
        processed_text = tokenize_text(transcription)
        return {"text": "".join(processed_text)}
