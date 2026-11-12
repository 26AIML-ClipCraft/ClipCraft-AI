"""Static server for the highlights page: no caching + HTTP Range (needed to play/seek video).  usage: python serve.py [port]"""
import http.server, os, re, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store"); super().end_headers()

    def send_head(self):
        rng = self.headers.get("Range"); path = self.translate_path(self.path)
        m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        a, b = m.group(1), m.group(2)
        start, end = (size - int(b), size - 1) if a == "" else (int(a), int(b) if b else size - 1)
        end = min(end, size - 1)
        if start > end:
            self.send_error(416); return None
        f = open(path, "rb"); f.seek(start); self._left = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path)); self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}"); self.send_header("Content-Length", str(self._left))
        self.end_headers(); return f

    def copyfile(self, src, dst):
        left = getattr(self, "_left", None); self._left = None
        try:
            while True:
                n = 65536 if left is None else min(65536, left)
                if n <= 0: break
                buf = src.read(n)
                if not buf: break
                dst.write(buf); left = None if left is None else left - len(buf)
        except (BrokenPipeError, ConnectionResetError):
            pass

http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 38417), H).serve_forever()
