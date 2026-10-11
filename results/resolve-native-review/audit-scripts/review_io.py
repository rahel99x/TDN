"""Read exact original bytes from verified canonical gzip storage."""
from functools import lru_cache
from pathlib import Path
import gzip,json
BASE=Path(__file__).resolve().parent
@lru_cache(maxsize=1)
def index():return json.loads((BASE/'evidence-index.json').read_bytes())
def names():return list(index()['files'])
def raw(name):
    entry=index()['files'][name]
    return gzip.decompress((BASE/'objects'/(entry['sha256']+'.gz')).read_bytes())
def load(name):return json.loads(raw(name))
def lines(name):
    entry=index()['files'][name]
    with gzip.open(BASE/'objects'/(entry['sha256']+'.gz'),'rt') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)
