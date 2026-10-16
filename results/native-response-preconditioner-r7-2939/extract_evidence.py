"""Validate and extract the uploaded R7 receipt; never execute archive contents."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
from zipfile import ZipFile

SHA256 = '9fafcb17b9afb5657319894fc0cb127851fe15ba054cc670dc3e53671b47bbe8'
RUN = 'launch-65e5bfaf1fd945f08fbfeb2726b2c29e-a1'

def extract(archive, output):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    root = Path(__file__).resolve().parents[2]
    if output.exists() or output == root or not output.is_relative_to(root):
        raise ValueError('Extraction requires a fresh directory inside this checkout')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError('Archive does not match the reviewed upload')
    with ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 2048 or sum(row.file_size for row in entries) > 256*1024**2:
            raise ValueError('Archive exceeds receipt bounds')
        names = set()
        for row in entries:
            path = PurePosixPath(row.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in row.filename
                    or not path.parts or path.parts[0] != RUN or row.filename in names
                    or row.flag_bits & 1 or row.is_dir()
                    or stat.S_IFMT(row.external_attr >> 16) not in (0, stat.S_IFREG)):
                raise ValueError('Unsupported archive entry: '+row.filename)
            names.add(row.filename)
        output.mkdir(parents=True, exist_ok=False)
        records = []
        for row in entries:
            target = output.joinpath(*PurePosixPath(row.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(row) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            records.append(dict(path=row.filename, bytes=target.stat().st_size,
                                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    result = dict(archive_sha256=digest, count=len(records), uncompressed_bytes=sum(r['bytes'] for r in records),
                  uploaded_code_executed=False, files=records)
    (output/'extraction-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive');parser.add_argument('output')
    args=parser.parse_args()
    result=extract(args.archive,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))
