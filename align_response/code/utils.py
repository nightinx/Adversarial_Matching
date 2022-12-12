import hashlib
import os
import sys
import urllib
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Optional, Union
import zipfile
import numpy as np
from tqdm import tqdm

# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
USER_AGENT = "pytorch/vision"


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def _calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    return md5 == _calculate_md5(fpath, **kwargs)


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def _check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def _save_response_content(
    content: Iterator[bytes],
    destination: str,
    length: Optional[int] = None,
) -> None:
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            pbar.update(len(chunk))


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def _urlretrieve(url: str, filename: str, chunk_size: int = 1024 * 32) -> None:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ) as response:
        _save_response_content(
            iter(lambda: response.read(chunk_size), b""),
            filename,
            length=response.length,
        )


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def _get_redirect_url(url: str, max_hops: int = 3) -> str:
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

    for _ in range(max_hops + 1):
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers)
        ) as response:
            if response.url == url or response.url is None:
                return url

            url = response.url
    else:
        raise RecursionError(
            f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect points to {url}."
        )


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def _download_url(
    url: str,
    root: str,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    max_redirect_hops: int = 3,
):
    fpath = os.path.expanduser(root)
    if filename:
        fpath = os.path.join(root, filename)

    url = _get_redirect_url(url, max_hops=max_redirect_hops)

    try:
        print("Downloading " + url + " to " + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, OSError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print(
                "Failed download. Trying https -> http instead. Downloading "
                + url
                + " to "
                + fpath
            )
            _urlretrieve(url, fpath)
        else:
            raise e

    if not _check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")

# def unzip(zip_path,read_file,write_path):
#     zipFile = zipfile.ZipFile(zip_path, 'r')

#     data = zipFile.read(read_file)

#     (lambda f, d: (f.write(d), f.close()))(open(write_path, 'w'), data)

#     zipFile.close()

def unzip(path_to_zip_file,directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)