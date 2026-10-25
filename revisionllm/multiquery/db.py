"""PostgreSQL + pgvector storage (Fig. 6): Video (1) -> (N) QueryCache."""
import numpy as np
import psycopg
from pgvector.psycopg import register_vector


def schema_sql(dim: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS video (
    video_id        BIGSERIAL PRIMARY KEY,
    video_hash      TEXT UNIQUE NOT NULL,          -- SHA-256 of the file
    vision_feature  TEXT,                          -- path of the CLIP frame features (.npy)
    audio_feature   TEXT,                          -- path of the CLAP features (.npy); NULL = no audio stream
    duration        REAL NOT NULL,                 -- seconds
    feature_version TEXT NOT NULL,                 -- extractor settings the tensors were made with
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS query_cache (
    query_id        BIGSERIAL PRIMARY KEY,
    video_id        BIGINT NOT NULL REFERENCES video(video_id) ON DELETE CASCADE,
    query_text      TEXT NOT NULL,
    query_embedding VECTOR({dim}) NOT NULL,
    start_time      REAL NOT NULL,                 -- seconds
    end_time        REAL NOT NULL,
    model_version   TEXT NOT NULL,                 -- checkpoint that produced the segment
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS query_cache_video_idx ON query_cache(video_id);
"""


class CacheDB:
    def __init__(self, dsn: str, embedding_dim: int = 768):
        self.dim = embedding_dim
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(schema_sql(embedding_dim))

    def close(self):
        self.conn.close()

    # ---- Video ----------------------------------------------------------------------------------------------
    def get_video(self, video_hash: str):
        row = self.conn.execute(
            "SELECT video_id, vision_feature, audio_feature, duration, feature_version FROM video WHERE video_hash=%s",
            (video_hash,)).fetchone()
        if row is None:
            return None
        return dict(video_id=row[0], vision_feature=row[1], audio_feature=row[2], duration=row[3],
                    feature_version=row[4])

    def put_video(self, video_hash, vision_feature, audio_feature, duration, feature_version) -> int:
        """Insert, or refresh the row of a known hash (stale feature_version) and drop its query cache."""
        with self.conn.transaction():
            row = self.conn.execute(
                """INSERT INTO video (video_hash, vision_feature, audio_feature, duration, feature_version)
                   VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT (video_hash) DO UPDATE SET vision_feature=EXCLUDED.vision_feature,
                       audio_feature=EXCLUDED.audio_feature, duration=EXCLUDED.duration,
                       feature_version=EXCLUDED.feature_version, created_at=now()
                   RETURNING video_id""",
                (video_hash, vision_feature, audio_feature, duration, feature_version)).fetchone()
            self.conn.execute("DELETE FROM query_cache WHERE video_id=%s", (row[0],))
        return row[0]

    # ---- QueryCache -------------------------------------------------------------------------------------------
    def search_query(self, video_id: int, embedding: np.ndarray, model_version: str):
        """Most similar cached query of THIS video (and this checkpoint). Returns (similarity, row) or None."""
        row = self.conn.execute(
            """SELECT query_id, query_text, start_time, end_time, 1 - (query_embedding <=> %s) AS sim
               FROM query_cache WHERE video_id=%s AND model_version=%s
               ORDER BY query_embedding <=> %s LIMIT 1""",
            (embedding, video_id, model_version, embedding)).fetchone()
        if row is None:
            return None
        return float(row[4]), dict(query_id=row[0], query_text=row[1], start_time=row[2], end_time=row[3])

    def put_query(self, video_id, query_text, embedding, start_time, end_time, model_version) -> int:
        return self.conn.execute(
            """INSERT INTO query_cache (video_id, query_text, query_embedding, start_time, end_time, model_version)
               VALUES (%s,%s,%s,%s,%s,%s) RETURNING query_id""",
            (video_id, query_text, embedding, start_time, end_time, model_version)).fetchone()[0]
