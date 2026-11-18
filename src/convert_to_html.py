import markdown
import re
import os

md_path = r'D:\btp\report\BTP_Final_Report.md'
html_path = r'D:\btp\report\BTP_Final_Report.html'

with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace carousel blocks with standard markdown images
text = re.sub(r'````carousel\s*', '', text)
text = re.sub(r'<!-- slide -->\s*', '', text)
text = re.sub(r'````\s*', '', text)

# Map relative images/ to absolute paths so HTML renderer can display them
img_dir = os.path.abspath(r'C:\Users\Shivam\.gemini\antigravity\brain\8948f3ed-82c9-453c-bc46-d82f7ce65d82\images').replace('\\', '/')
text = text.replace('images/', img_dir + '/')

# Convert markdown to HTML
html_body = markdown.markdown(text, extensions=['extra', 'tables', 'fenced_code', 'toc'])

css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono&display=swap');
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #1a202c;
    background-color: #ffffff;
    padding: 40px;
    max-width: 960px;
    margin: auto;
}
h1 { font-size: 26px; color: #0f172a; border-bottom: 2px solid #0284c7; padding-bottom: 8px; margin-top: 30px; }
h2 { font-size: 20px; color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-top: 25px; }
h3 { font-size: 16px; color: #334155; margin-top: 20px; }
table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 13px;
}
th, td {
    padding: 10px 12px;
    border: 1px solid #cbd5e1;
    text-align: left;
}
th {
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 600;
}
tr:nth-child(even) { background-color: #f8fafc; }
img {
    max-width: 90%;
    height: auto;
    display: block;
    margin: 20px auto;
    border-radius: 6px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
em {
    display: block;
    text-align: center;
    font-size: 12px;
    color: #64748b;
    margin-top: -10px;
    margin-bottom: 20px;
}
pre {
    background: #0f172a;
    color: #f8fafc;
    padding: 15px;
    border-radius: 6px;
    overflow-x: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
blockquote {
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
    margin: 15px 0;
    padding: 12px 18px;
    border-radius: 0 4px 4px 0;
    font-size: 13px;
}
@media print {
    body { padding: 15px; }
    h1, h2 { page-break-after: avoid; }
    table, img, pre { page-break-inside: avoid; }
}
</style>
"""

full_html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BTP Final Report - ML Driven RHEA Design</title>
""" + css + """
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
""" + html_body + """
</body>
</html>
"""

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(full_html)

print("HTML conversion successful:", html_path)
