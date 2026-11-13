"""Shared by the run_*.py scripts: repo-root setup, the title rule, and the editing service."""
import glob, os, re, sys, unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def title_of(path: str) -> str:
    """Video title from the file name (the text query of UniVTG)."""
    t = unicodedata.normalize("NFC", os.path.splitext(os.path.basename(path))[0])
    t = re.sub(r"^YTDown\.com_YouTube_Media_[\w-]+_", "", t)
    return re.sub(r"[…@＂\"\[\]]", " ", t).strip()


def videos():
    """Every *.mp4 in the repo root, sorted (the number in the output file names is the position in this list)."""
    return sorted(glob.glob("*.mp4"))


def load_openai_key():
    """OPENAI_API_KEY from the environment, else from ~/.openai_key (never printed)."""
    key = os.path.expanduser("~/.openai_key")
    if "OPENAI_API_KEY" not in os.environ and os.path.exists(key):
        os.environ["OPENAI_API_KEY"] = open(key).read().strip()


def make_service():
    from revisionllm.editing.config import EditConfig
    from revisionllm.editing.db import EditDB
    from revisionllm.editing.scorers import make_scorer
    from revisionllm.editing.service import EditingService
    from revisionllm.multiquery.config import CacheConfig
    cc = CacheConfig()
    db = EditDB(cc.dsn, cc.embedding_dim)
    return EditingService(cc, EditConfig(), "checkpoints/clip/ViT-L-14.pt", db=db, scorer=make_scorer(), grounder=None, query_gen=None)
