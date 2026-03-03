# Contracts

## FeatureVector v1
- File: `feature_vector_v1.schema.json`
- Shape: `[410]`
- Dtype: `float32`
- Produced by: `Backend/backend_app/core/picam_feature_extractor.py`

## SequenceInput v1
- File: `sequence_input_v1.schema.json`
- Shape: `[1, 14, 410]`
- Dtype: `float32`
- Consumed by: `Train/model.onnx` and runtime classifier backends

## Runtime metadata
The runtime model uses `Train/model.runtime.json` with fields:
- `labels`
- `num_classes`
- `input_size`
- `sequence_length`
- `feature_contract_version`

Schema: `runtime_metadata.schema.json`
