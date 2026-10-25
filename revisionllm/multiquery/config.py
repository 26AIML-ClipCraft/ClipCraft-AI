import os
from dataclasses import dataclass, field


@dataclass
class CacheConfig:
    """Settings of the cache layer. The values that change what is stored are folded into the version strings,
    so a cache written with other settings or another checkpoint is never reused silently."""
    dsn: str = field(default_factory=lambda: os.environ.get("CLIPCRAFT_DB_DSN", "postgresql://localhost:5432/clipcraft"))
    cache_dir: str = "data/multiquery_cache"          # vision.npy / audio.npy per video hash
    similarity_threshold: float = 0.95                # cosine similarity; hit only when strictly greater
    embedding_dim: int = 768                          # CLIP ViT-L/14 text EOT feature (the model's own query_cls_feats)
    # fixed by the released pipeline (scripts/audio/env.sh); part of feature_version
    clip_name: str = "ViT-L-14"
    clap_name: str = "laion/clap-htsat-fused"
    feature_fps: float = 2.0
    audio_hop_sec: float = 1.0
    audio_dim: int = 512
    # features are extracted at a different time/model setting -> bump to invalidate every stored tensor
    feature_schema: int = 1

    @property
    def feature_version(self) -> str:
        return (f"clip={self.clip_name}@{self.feature_fps:g}fps|clap={self.clap_name}@{self.audio_hop_sec:g}s"
                f"|schema={self.feature_schema}")
