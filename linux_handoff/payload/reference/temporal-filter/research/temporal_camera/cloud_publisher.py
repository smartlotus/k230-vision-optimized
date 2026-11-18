"""Reuse the accepted K230 cloud publisher with CLI OAuth and RoadNet-local records."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
LEGACY=Path("C:/Users/28389/Desktop/k230-vision-optimized")
sys.path.insert(0,str(LEGACY/"research/topface"))
from publisher import control
from cloud_store import API,atomic_json


def source_url(prep,file_path,token_file,source_notebook):
    path=Path(source_notebook).resolve()
    if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=prep["source_sha256"]:
        raise ValueError("RoadNet frozen notebook source mismatch")
    expected=json.loads(path.read_text(encoding="utf8"))
    def cells(notebook):
        return [(c["cell_type"],"".join(c["source"]) if isinstance(c["source"],list) else c["source"])
                for c in notebook["cells"]]
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiGetKernelSessionStatusRequest,ApiListKernelSessionOutputRequest
    api=KaggleApi()
    api.authenticate()
    owner,slug=prep["ref"].split("/")
    with api.build_kaggle_client() as client:
        get=ApiGetKernelRequest()
        get.user_name,get.kernel_slug,get.version_label=owner,slug,f"v{prep['version']}"
        saved=client.kernels.kernels_api_client.get_kernel(get)
        if saved.metadata.id!=prep["kernel_id"] or not saved.metadata.is_private:
            raise ValueError("Saved source ID/privacy differs")
        if cells(json.loads(saved.blob.source))!=cells(expected):
            raise ValueError("Saved source cells differ from frozen notebook")
        req=ApiGetKernelSessionStatusRequest()
        req.user_name,req.kernel_slug,req.version_label=owner,slug,f"v{prep['version']}"
        if client.kernels.kernels_api_client.get_kernel_session_status(req).status.name!="COMPLETE":
            raise ValueError("Exact saved source is not complete")
        page=None
        for _ in range(100):
            req=ApiListKernelSessionOutputRequest()
            req.user_name,req.kernel_slug,req.version_label=owner,slug,f"v{prep['version']}"
            req.page_size,req.page_token=100,page
            response=client.kernels.kernels_api_client.list_kernel_session_output(req)
            found=[f for f in response.files if f.file_name==file_path]
            if found:
                if len(found)!=1:
                    raise ValueError("Ambiguous saved archive")
                control.source_endpoint(found[0].url)
                return found[0].url
            page=response.next_page_token
            if not page:
                break
    raise ValueError("Original saved archive missing")


def folder_for(run_id):
    if not re.fullmatch(r"k230-[a-z0-9-]{8,100}",run_id):
        raise ValueError("Safe K230 run identity required")
    return ROOT/"docs/temporal_camera/cloud/publisher"/run_id


def protected_path(run_id):
    folder_for(run_id)
    path=ROOT/".ops/local/temporal-publisher-private"/(run_id+".dpapi")
    if subprocess.run(["git","-C",str(ROOT),"check-ignore","--quiet",str(path)]).returncode:
        raise ValueError("Encrypted recovery path must remain Git ignored")
    return path


control.source_url=source_url
control.folder_for=folder_for
control.protected_path=protected_path


def admit(run_id, input_sha, out):
    control.authorized()
    drive=control.drive_local()
    prior_root=LEGACY/"docs/topface/v11/cloud/publisher/k230-topface-v11-formal-a01"
    prior=json.loads((prior_root/"TRANSFER_RECEIPT.json").read_text(encoding="utf8"))
    binding=json.loads((prior_root/"BINDING.json").read_text(encoding="utf8"))
    if prior["status"]!="DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED" or prior["secret_cleanup"]!="CONFIRMED_DELETED":
        raise ValueError("Original independent publisher qualification is incomplete")
    source_commit=binding["publisher_source_commit"]
    source=control.command("git","show",source_commit+":research/topface/publisher/ci_transfer.py")
    if source!=(LEGACY/"research/topface/publisher/ci_transfer.py").read_text(encoding="utf8").strip():
        raise ValueError("Publisher source drift")
    name=run_id+"-RUN-ADMISSION.json"
    with drive.request("GET",API,params={"q":f"'{control.FOLDER}' in parents and name = '{name}' and trashed = false",
                                       "fields":"files(id,name,parents,appProperties)"}) as response:
        existing=response.json().get("files",[])
    if out.exists():
        record=json.loads(out.read_text(encoding="utf8"))
        if len(existing)!=1 or existing[0]["id"]!=record["drive_marker_id"] or record["input_manifest_sha256"]!=input_sha:
            raise ValueError("Original admission requires reconciliation")
        return record
    if existing:
        if len(existing)!=1 or existing[0].get("appProperties",{}).get("input_sha256")!=input_sha:
            raise ValueError("Ambiguous prior admission")
        marker=existing[0]["id"]
    else:
        attempt=out.with_suffix(".attempt.json")
        if attempt.exists():
            raise ValueError("Original admission attempt requires reconciliation")
        atomic_json(attempt,{"run_id":run_id,"input_manifest_sha256":input_sha,
                             "folder_id":control.FOLDER,"automatic_retry":False})
        marker=drive.reserve_run(run_id,input_sha)
    record={"status":"INDEPENDENT_PUBLISHER_ADMITTED","run_id":run_id,
            "input_manifest_sha256":input_sha,"drive_marker_id":marker,"folder_id":control.FOLDER,
            "publisher_source_commit":source_commit,"automatic_replay":False,
            "prior_route_receipt_sha256":hashlib.sha256((prior_root/"TRANSFER_RECEIPT.json").read_bytes()).hexdigest(),
            "scope":"Existing user-authorized same-scope K230 cloud-to-cloud publication",
            "observed_utc":datetime.now(timezone.utc).isoformat()}
    atomic_json(out,record)
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("admit")
    a.add_argument("run_id")
    a.add_argument("input_sha")
    a.add_argument("out",type=Path)
    pub=sub.add_parser("publish")
    pub.add_argument("frozen",type=Path)
    pub.add_argument("observation",type=Path)
    pub.add_argument("result",type=Path)
    pub.add_argument("admission",type=Path)
    args=p.parse_args()
    try:
        if args.action=="admit":
            value=admit(args.run_id,args.input_sha,args.out)
        else:
            from publisher.publish_completed import publish
            record=json.loads(args.admission.read_text(encoding="utf8"))
            value=publish(args.frozen,args.observation,args.result,None,record["publisher_source_commit"])
        print(json.dumps(value),flush=True)
    except Exception as error:
        print(json.dumps({"status":"RECONCILE_ORIGINAL_PUBLICATION","error_type":type(error).__name__}),flush=True)
        raise SystemExit(1) from None
