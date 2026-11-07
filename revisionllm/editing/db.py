"""Results of the editing features, stored next to multiquery's video / query_cache tables: Video (1) -> (N) edit_result."""
import json

from psycopg.types.json import Jsonb

from revisionllm.multiquery.db import CacheDB

EDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS edit_result (
    result_id   BIGSERIAL PRIMARY KEY,
    video_id    BIGINT NOT NULL REFERENCES video(video_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,                   -- 'summary' | 'caption' | 'scenario' | 'highlight'
    params_key  TEXT NOT NULL,                   -- hash of settings + feature_version + head version (see service.params_key)
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, kind, params_key)
);
"""


class EditDB(CacheDB):
    def __init__(self, dsn: str, embedding_dim: int = 768):
        super().__init__(dsn, embedding_dim)
        self.conn.execute(EDIT_SCHEMA)

    def get_result(self, video_id: int, kind: str, params_key: str):
        row = self.conn.execute("SELECT payload FROM edit_result WHERE video_id=%s AND kind=%s AND params_key=%s",
                                (video_id, kind, params_key)).fetchone()
        return None if row is None else row[0]

    def put_result(self, video_id: int, kind: str, params_key: str, payload) -> None:
        self.conn.execute(
            """INSERT INTO edit_result (video_id, kind, params_key, payload) VALUES (%s,%s,%s,%s)
               ON CONFLICT (video_id, kind, params_key) DO UPDATE SET payload=EXCLUDED.payload, created_at=now()""",
            (video_id, kind, params_key, Jsonb(json.loads(json.dumps(payload)))))
