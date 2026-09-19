"""
ClipCraft data pipeline for the audio-visual ReVisionLLM experiment.

  subsets.py               - sample Train-sub / Val-sub / Test-sub from the official VidChapters-7M splits
  download_and_extract.py  - yt-dlp -> CLIP (2 fps) + CLAP (1 s hop) -> LMDB, rolling delete, resumable
  extract_clip.py          - CLIP ViT-L/14 frame features from a video file
  extract_clap.py          - CLAP audio features from a video / audio file
  finalize_subsets.py      - keep only ids present in BOTH LMDBs, freeze the final lists, overlap check
  label_query_types.py     - keyword + LLM labelling of "audio cue present" queries (H2 breakdown)
  lmdb_utils.py            - shared LMDB writer/reader helpers
"""
