"""Download the public release checkpoint and verify its SHA-256."""
import hashlib
from pathlib import Path
import urllib.request

URL='https://github.com/shuke-dsj/microfluidic-oil-segmentation/releases/download/v1.0.0/attention_resunet.pth'
SHA256='ceec24215bdc38dc0a3cffc34cc6f43a9e25701635a0d2847441f92aa2bc4d12'


def main():
    output=Path(__file__).resolve().parents[1]/'weights/attention_resunet.pth'
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        if hashlib.sha256(output.read_bytes()).hexdigest()==SHA256:
            print(f'Already verified: {output}')
            return
        raise RuntimeError('Existing checkpoint differs; move it before downloading')
    temporary=output.with_suffix('.download')
    try:
        urllib.request.urlretrieve(URL,temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=SHA256:
            raise RuntimeError('Checkpoint checksum mismatch')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Verified: {output}')


if __name__=='__main__':
    main()
