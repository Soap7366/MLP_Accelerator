from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parents[1]
m=json.loads((r/'sha256.json').read_text())
for n,h in m.items():
    assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
d=json.loads((r/'release.json').read_text())
assert d['candidate']=='r108_readonly'
for n in ('mlp.bit','mlp.hwh'):
    assert hashlib.sha256((r/'ps/r108_readonly'/n).read_bytes()).hexdigest()==d[n]
print('Final read-only release verified:',len(m),'files')
