"""The scorer that produces the saliency curve of the editing module: released UniVTG (CLIP-B/32 checkpoint, frozen) on the cached
CLIP-L/14 features through the aligned Linear Adapter 768->512 (checkpoints/univtg/). Query = video title text.
See checkpoints/univtg/README.md."""


def make_scorer():
    from revisionllm.editing.univtg.scorer import UniVTGScorer
    return UniVTGScorer()
