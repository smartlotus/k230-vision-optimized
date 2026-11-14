"""Cloud-side Drive persistence. No credentials are embedded or logged."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
from urllib.parse import urlsplit
import requests

SECRET_NAME = "DOASTRA_GDRIVE_OAUTH_JSON"
API = "https://www.googleapis.com/drive/v3/files"
CHUNK = 8*1024*1024


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".part")
    temporary.write_text(json.dumps(value, indent=2)+"\n", encoding="utf8")
    temporary.replace(path)


class Drive:
    def __init__(self, folder_id):
        if not folder_id or not folder_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Explicit Google Drive destination folder required")
        raw = os.environ.get(SECRET_NAME)
        if raw is None:
            from kaggle_secrets import UserSecretsClient
            raw = UserSecretsClient().get_secret(SECRET_NAME)
        self.oauth = json.loads(raw)
        self.folder_id = folder_id
        self.session = requests.Session()
        self.token, self.expiry = "", 0.

    def request(self, method, url, **kwargs):
        host = urlsplit(url)
        if host.scheme != "https" or host.netloc != "www.googleapis.com":
            raise ValueError("Unexpected Drive endpoint")
        if time.time()+120 >= self.expiry:
            response = self.session.post("https://oauth2.googleapis.com/token", data={
                "client_id": self.oauth["client_id"],
                "client_secret": self.oauth["client_secret"],
                "refresh_token": self.oauth["refresh_token"],
                "grant_type": "refresh_token"}, timeout=(15, 45), allow_redirects=False)
            response.raise_for_status()
            payload = response.json()
            self.token = payload["access_token"]
            self.expiry = time.time()+payload["expires_in"]
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = "Bearer "+self.token
        response = self.session.request(method, url, headers=headers,
            timeout=(15, 90), allow_redirects=False, **kwargs)
        if response.is_redirect and response.status_code != 308:
            response.close()
            raise RuntimeError("Unexpected Google Drive redirect")
        response.raise_for_status()
        return response

    def verify_folder(self):
        with self.request("GET", API+"/"+self.folder_id, params={
                "fields": "id,mimeType,trashed,capabilities(canAddChildren)"}) as response:
            folder = response.json()
        if (folder.get("id") != self.folder_id or folder.get("trashed") or
                folder.get("mimeType") != "application/vnd.google-apps.folder" or
                folder.get("capabilities", {}).get("canAddChildren") is not True):
            raise RuntimeError("Expected writable K230 destination folder not verified")

    def export(self, path, run_id, receipt_path):
        path = Path(path)
        digest, size = sha(path), path.stat().st_size
        remote_name = f"{run_id}-{digest[:16]}{''.join(path.suffixes)}"
        # The session journal is outside the archived result. Retain for reconciliation.
        journal_path = Path("/tmp/k230-private-upload")/(run_id+".upload.json")
        if journal_path.exists():
            journal = json.loads(journal_path.read_text())
            if journal["sha256"] != digest or journal["name"] != remote_name:
                raise RuntimeError("Existing upload belongs to different artifact bytes")
        else:
            # Reconcile an existing immutable object before creating another one.
            with self.request("GET", API, params={"q":
                    f"'{self.folder_id}' in parents and name = '{remote_name}' and trashed = false",
                    "fields": "files(id,name,size,parents,sha256Checksum)"}) as response:
                files = response.json().get("files", [])
            if len(files) > 1:
                raise RuntimeError("Ambiguous duplicate Drive archive name")
            if files:
                journal = {"name": remote_name, "sha256": digest, "metadata": files[0]}
            else:
                with self.request("POST", "https://www.googleapis.com/upload/drive/v3/files",
                    params={"uploadType": "resumable", "fields": "id,name,size,parents,sha256Checksum"},
                    headers={"X-Upload-Content-Length": str(size),
                             "X-Upload-Content-Type": "application/octet-stream"},
                    json={"name": remote_name, "parents": [self.folder_id],
                          "appProperties": {"project": "k230-topface", "sha256": digest}}) as response:
                    location = response.headers["Location"]
                journal = {"name": remote_name, "sha256": digest, "location": location}
            atomic_json(journal_path, journal)
        if "metadata" not in journal:
            location = journal["location"]
            with self.request("PUT", location, headers={
                    "Content-Length": "0", "Content-Range": f"bytes */{size}"}, data=b"") as response:
                if response.status_code in (200, 201):
                    journal["metadata"] = response.json()
                    offset = size
                else:
                    acknowledged = response.headers.get("Range", "")
                    offset = int(acknowledged.split("-")[-1])+1 if acknowledged else 0
            with path.open("rb") as stream:
                while offset < size:
                    stream.seek(offset)
                    data = stream.read(min(CHUNK, size-offset))
                    with self.request("PUT", location, data=data, headers={
                            "Content-Length": str(len(data)),
                            "Content-Range": f"bytes {offset}-{offset+len(data)-1}/{size}"}) as response:
                        if response.status_code in (200, 201):
                            journal["metadata"] = response.json()
                            offset = size
                        else:
                            acknowledged = response.headers.get("Range", "")
                            confirmed = int(acknowledged.split("-")[-1])+1 if acknowledged else 0
                            if not offset < confirmed <= offset+len(data):
                                raise RuntimeError("Drive upload failed to advance")
                            offset = confirmed
            atomic_json(journal_path, journal)
        meta = journal["metadata"]
        if (meta.get("name") != remote_name or int(meta.get("size", -1)) != size or
                meta.get("parents") != [self.folder_id]):
            raise RuntimeError("Drive archive metadata mismatch")
        returned_size, returned_hash = 0, hashlib.sha256()
        with self.request("GET", API+"/"+meta["id"], params={"alt": "media"}, stream=True) as response:
            for chunk in response.iter_content(CHUNK):
                if chunk:
                    returned_size += len(chunk)
                    returned_hash.update(chunk)
        if (returned_size, returned_hash.hexdigest()) != (size, digest):
            raise RuntimeError("Drive full byte readback mismatch")
        receipt = {"schema": "k230-cloud-direct-write-v1", "run_id": run_id,
            "status": "FULL_BYTE_SHA256_PASS", "folder_id": self.folder_id,
            "file_id": meta["id"], "name": remote_name, "bytes": size,
            "sha256": digest, "automatic_replay": False}
        atomic_json(receipt_path, receipt)
        return receipt


def archive_directory(root, target):
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    manifest = [{"name": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                 "sha256": sha(p)} for p in files]
    atomic_json(root/"MEMBERS.json", manifest)
    with tarfile.open(target, "w:gz") as output:
        for path in files+[root/"MEMBERS.json"]:
            if path.is_symlink():
                raise ValueError("Symlinks cannot enter cloud archive")
            output.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    return manifest
