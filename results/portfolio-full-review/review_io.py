"""Read exact-byte, hash-verified canonical evidence in compressed local storage."""
from pathlib import Path
import gzip
import json

BASE = Path(__file__).resolve().parent

def index():
    return json.loads((BASE / 'evidence-index.json').read_bytes())

def names():
    return list(index()['files'])

def raw(name):
    entry = index()['files'][name]
    return gzip.decompress((BASE / 'objects' / (entry['sha256'] + '.gz')).read_bytes())

def load(name):
    return json.loads(raw(name))

def lines(name):
    entry = index()['files'][name]
    with gzip.open(BASE / 'objects' / (entry['sha256'] + '.gz'), 'rt') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)
