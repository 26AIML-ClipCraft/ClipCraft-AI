"""Draws the highlight and summary pipelines (as implemented) to SVG; render to PNG with a headless browser."""
import html

FILL = dict(white="#ffffff", teal="#E3F1EA", yellow="#FCF3D3", purple="#ECE8F8", blue="#E6EEF9", green="#F0F7F0",
            orange="#FBEBDD", out="#ffffff")
INK, STROKE = "#2F3A4C", "#3B4558"
GREEN_T = "#2F7D5B"


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.p, self.cells = w, h, [], []

    def _t(self, x, y, s, size, bold=False, color=INK):
        self.p.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" font-weight="{700 if bold else 400}" '
                      f'fill="{color}">{html.escape(s)}</text>')

    def box(self, x, y, w, h, title, sub=(), fill="white"):
        _sub = [sub] if isinstance(sub, str) else list(sub)
        val = f"<b style=\"font-size:16px\">{html.escape(title)}</b>" + "".join(f"<br><span style=\"font-size:12px;color:#4B566A\">{html.escape(t)}</span>" for t in _sub)
        self.cells.append(("v", x, y, w, h, val, f"rounded=0;whiteSpace=wrap;html=1;fillColor={FILL[fill]};strokeColor={STROKE};strokeWidth=2;fontColor={INK};"))
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{FILL[fill]}" stroke="{STROKE}" stroke-width="2"/>')
        sub = [sub] if isinstance(sub, str) else list(sub)
        lines = 1 + len(sub)
        total = 26 + 22 * len(sub)
        y0 = y + (h - total) / 2 + 22
        self._t(x + w / 2, y0, title, 23, True)
        for i, s in enumerate(sub):
            self._t(x + w / 2, y0 + 28 + 22 * i, s, 17, False, "#4B566A")

    def group(self, x, y, w, h, label, right=False):
        self.cells.append(("v", x, y, w, h, f"<b>{html.escape(label)}</b>", f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor={FILL['green']};strokeColor=none;fontColor={GREEN_T};fontSize=14;verticalAlign=top;align={'right' if right else 'left'};spacingLeft=16;spacingRight=16;spacingTop=6;"))
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="26" fill="{FILL["green"]}"/>')
        tx, anchor = (x + w - 24, "end") if right else (x + 24, "start")
        self.p.append(f'<text x="{tx}" y="{y + 36}" text-anchor="{anchor}" font-size="22" font-weight="700" fill="{GREEN_T}">{html.escape(label)}</text>')

    def arrow(self, pts, dot=False):
        self.cells.append(("e", pts, True))
        d = " ".join(f"{a},{b}" for a, b in pts)
        self.p.append(f'<polyline points="{d}" fill="none" stroke="{STROKE}" stroke-width="2.5" marker-end="url(#ah)"/>')
        if dot:
            self.p.append(f'<circle cx="{pts[0][0]}" cy="{pts[0][1]}" r="4" fill="{STROKE}"/>')

    def line(self, pts):
        self.cells.append(("e", pts, False))
        d = " ".join(f"{a},{b}" for a, b in pts)
        self.p.append(f'<polyline points="{d}" fill="none" stroke="{STROKE}" stroke-width="2.5"/>')

    def dot(self, x, y):
        self.cells.append(("v", x - 4, y - 4, 8, 8, "", f"ellipse;whiteSpace=wrap;html=1;fillColor={STROKE};strokeColor={STROKE};"))
        self.p.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{STROKE}"/>')

    def text(self, x, y, s, size=18, color="#4B566A", anchor="middle"):
        self.p.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}">{html.escape(s)}</text>')

    def save_drawio(self, path, name="pipeline"):
        """draw.io file (File > Open) / mxGraphModel xml (Extras > Edit Diagram)."""
        out, n = [], 2
        for c in self.cells:
            n += 1
            if c[0] == "v":
                _, x, y, w, h, val, st = c
                out.append(f'<mxCell id="{n}" value="{html.escape(val, quote=True)}" style="{st}" vertex="1" parent="1">'
                           f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
            else:
                _, pts, arr = c
                st = f"edgeStyle=none;html=1;rounded=0;strokeColor={STROKE};strokeWidth=2.5;endArrow={'block' if arr else 'none'};endFill=1;"
                mid = "".join(f'<mxPoint x="{a}" y="{b}"/>' for a, b in pts[1:-1])
                arr_xml = f'<Array as="points">{mid}</Array>' if mid else ""
                out.append(f'<mxCell id="{n}" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                           f'<mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/><mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/>{arr_xml}</mxGeometry></mxCell>')
        model = (f'<mxGraphModel dx="1200" dy="900" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" '
                 f'pageScale="1" pageWidth="{self.w}" pageHeight="{self.h}" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                 + "".join(out) + "</root></mxGraphModel>")
        open(path, "w", encoding="utf-8").write(f'<mxfile host="app.diagrams.net"><diagram name="{name}" id="p1">{model}</diagram></mxfile>')
        open(path.replace(".drawio", "_model.xml"), "w", encoding="utf-8").write(model)

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
                f'font-family="\'Noto Sans KR\', \'Helvetica Neue\', Arial, sans-serif"><rect width="100%" height="100%" fill="#fff"/>'
                f'<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{STROKE}"/></marker></defs>')
        open(path, "w", encoding="utf-8").write(head + "".join(self.p) + "</svg>")


def highlight():
    """Default scorer: UniVTG (frozen) on the cached L/14 features through the trained Linear Adapter."""
    s = Svg(1500, 1570)
    cx = 750
    s.text(cx, 1535, "W = 12 s (< 6 min), 20 s (< 20 min), 30 s (longer): the longest a clip may grow (same value as the summary window)", 16)
    s.box(550, 30, 400, 100, "Highlight Clips", ["greedy peak picking, 3 s up to window W,", "gap ≥ 2 s, total ≈ 10 % (15–60 s)"])
    s.box(575, 190, 350, 70, "Highlight Score", "per second, 0–1")
    s.box(480, 320, 540, 120, "Weighted Late Fusion", ["0.7 S + 0.3 A", "each term rank-normalised, smoothed 3 s"], "yellow")
    s.arrow([(cx, 190), (cx, 130)]); s.arrow([(cx, 320), (cx, 260)])
    s.box(570, 520, 360, 90, "Saliency Score (S)", "UniVTG saliency (clip repeated to 1 s)", "teal")
    s.box(1080, 520, 360, 90, "Audio Score (A)", [], "teal")
    s.line([(1260, 520), (1260, 480), (cx, 480)]); s.dot(cx, 480)
    s.arrow([(cx, 520), (cx, 440)])
    # audio branch
    s.box(1080, 700, 360, 130, "Audio Event Scoring", ["CLAP audio vs. text prompts, softmax", "laughter 1 · applause 1 · music .6", "speech .5 · ambient .1 · silence −.5"], "purple")
    s.arrow([(1260, 700), (1260, 610)])
    # saliency branch
    s.group(440, 650, 620, 660, "UniVTG Saliency Branch", right=True)
    s.box(520, 710, 460, 90, "Similarity Score", "cosine(video token, pooled query)", "teal")
    s.arrow([(cx, 710), (cx, 610)])
    s.box(480, 870, 220, 90, "Feed Forward", ["frozen"], "blue")
    s.box(800, 870, 220, 90, "Att. Pooler", ["frozen"], "purple")
    s.arrow([(590, 870), (590, 800)]); s.arrow([(910, 870), (910, 800)])
    s.box(480, 1030, 220, 100, "Linear Adapter", ["768 → 512 (trained)"], "orange")
    s.box(800, 1030, 220, 100, "Feed Forward", ["frozen"], "blue")
    s.arrow([(590, 1030), (590, 960)]); s.arrow([(910, 1030), (910, 960)])
    s.box(480, 1190, 220, 90, "2-s Clip Pooling", ["mean of 2 s, L2"])
    s.box(800, 1190, 220, 90, "Text Encoder", ["CLIP-B/32, frozen"], "purple")
    s.arrow([(590, 1190), (590, 1130)]); s.arrow([(910, 1190), (910, 1130)])
    # inputs
    s.box(60, 1380, 360, 100, "Visual Features", "CLIP ViT-L/14, 768-d, 1 s (cached)")
    s.box(780, 1380, 260, 100, "Video Title", "file name")
    s.box(1080, 1380, 360, 100, "Audio Features", "CLAP, 512-d, 1 s (cached)")
    s.line([(240, 1380), (240, 1335), (590, 1335)]); s.arrow([(590, 1335), (590, 1280)])
    s.arrow([(910, 1380), (910, 1280)])
    s.arrow([(1260, 1380), (1260, 830)])
    return s


def summary():
    s = Svg(1500, 1680)
    s.box(480, 30, 540, 100, "Editing Scenario", ["title, summary, scenes (chronological)", "each with a description and a reason"])
    s.box(480, 190, 540, 120, "Scenario Recommendation (GPT)", ["captions + times → JSON", "missing scenes dropped, sorted by time"], "yellow")
    s.arrow([(750, 190), (750, 130)])
    s.box(480, 370, 540, 110, "Segment Captions (GPT)", ["what is visible in each segment,", "from its key frames"], "purple")
    s.arrow([(750, 370), (750, 310)])
    s.box(480, 540, 540, 120, "Key Frames", ["≈ 1 per 8 s (2–5) + importance peak", "1024 px, detail = high"], "blue")
    s.arrow([(750, 540), (750, 480)])
    s.box(480, 720, 540, 130, "MMR Selection", ["0.7 · importance − 0.3 · similarity to picked", "8 segments, returned in time order"], "yellow")
    s.arrow([(750, 720), (750, 660)])
    s.box(170, 920, 440, 120, "Segment Importance (I)", ["per W-s window: mean of the top", "half of the saliency, rank-normalised"], "teal")
    s.box(890, 920, 440, 120, "Segment Content Vector", ["per W-s window:", "mean of [ CLIP ; 0.5 · CLAP ]"], "teal")
    s.arrow([(390, 920), (390, 880), (620, 880), (620, 850)])
    s.arrow([(1110, 920), (1110, 880), (880, 880), (880, 850)])
    s.box(60, 1110, 360, 120, "Saliency Score (S)", ["UniVTG saliency only", "(no audio)"], "teal")
    s.arrow([(260, 1110), (260, 1040)])
    s.box(60, 1290, 500, 130, "UniVTG + Adapter", ["2-s pooling → Linear Adapter 768→512 →", "frozen UniVTG saliency (cosine with title)"], "blue")
    s.arrow([(260, 1290), (260, 1230)])
    s.box(60, 1530, 240, 90, "Video Title", "file name")
    s.box(520, 1530, 280, 90, "Visual Features", "CLIP ViT-L/14, 768-d")
    s.box(1060, 1530, 280, 90, "Audio Features", "CLAP, 512-d")
    s.arrow([(180, 1530), (180, 1420)])
    s.dot(660, 1480); s.line([(660, 1480), (440, 1480)]); s.arrow([(440, 1480), (440, 1420)])
    s.line([(660, 1530), (660, 1480)])
    s.arrow([(660, 1480), (1050, 1480), (1050, 1040)])        # visual -> content vector
    s.arrow([(1200, 1530), (1200, 1040)])                      # audio  -> content vector
    s.text(750, 1660, "W = 12 s (video < 6 min), 20 s (< 20 min), 30 s (longer)", 16)
    return s


if __name__ == "__main__":
    import os
    d = os.path.dirname(os.path.abspath(__file__))
    for name, fn in (("highlight_pipeline", highlight), ("summary_pipeline", summary)):
        g = fn()
        g.save(os.path.join(d, name + ".svg"))
        g.save_drawio(os.path.join(d, name + ".drawio"), name)
