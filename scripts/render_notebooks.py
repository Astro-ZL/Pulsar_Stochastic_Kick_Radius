"""Static standalone HTML rendering of standard notebook JSON and saved outputs."""
from pathlib import Path
import html,json
ROOT=Path(__file__).resolve().parents[1]
def render(path):
    nb=json.loads(path.read_text());parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+html.escape(path.stem)+'</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#17212b}pre{white-space:pre-wrap;background:#f3f5f7;padding:18px;border-radius:6px;overflow:auto}section{margin:24px 0}img{max-width:100%}.markdown{white-space:pre-wrap;background:#fff}small{color:#667}</style><h1>'+html.escape(path.stem)+'</h1><small>Notebook with saved outputs.</small>']
    for c in nb['cells']:
        src=''.join(c['source']);parts.append('<section>')
        parts.append('<pre class="'+('markdown' if c['cell_type']=='markdown' else 'code')+'">'+html.escape(src)+'</pre>')
        for o in c.get('outputs',[]):
            if o.get('output_type')=='stream':parts.append('<pre>'+html.escape(''.join(o.get('text',[])))+'</pre>')
            d=o.get('data',{})
            if 'image/png' in d:parts.append('<img alt="Recorded figure output" src="data:image/png;base64,'+''.join(d['image/png'])+'">')
            elif 'text/plain' in d:parts.append('<pre>'+html.escape(''.join(d['text/plain']))+'</pre>')
        parts.append('</section>')
    out=ROOT/'notebooks/html'/path.with_suffix('.html').name;out.parent.mkdir(exist_ok=True);out.write_text(''.join(parts))
if __name__=='__main__':
    for p in sorted((ROOT/'notebooks').glob('*.ipynb')):render(p);print(p.name)
