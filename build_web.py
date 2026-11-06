"""Inline core.js and data.js into the page template -> web/index.html (single self-contained file)."""
from pathlib import Path
W = Path(__file__).parent / "web"
html = (W / "template.html").read_text()
html = html.replace("<script>/*CORE*/</script>", "<script>\n" + (W / "core.js").read_text() + "\n</script>")
html = html.replace("<script>/*DATA*/</script>", "<script>\n" + (W / "data.js").read_text() + "\n</script>")
(W / "index.html").write_text(html)
print("index.html MB", round(len(html) / 1e6, 2))
