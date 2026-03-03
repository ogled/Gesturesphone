from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureVectorContract:
    version: str = "v1"
    dtype: str = "float32"
    sequence_length: int = 14
    feature_size: int = 410
    motion_energy_index: int = 379


FEATURE_CONTRACT = FeatureVectorContract()

