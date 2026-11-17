"""Single-send Kaggle operations using the current CLI OAuth cache."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from kaggle.api.kaggle_api_extended import KaggleApi


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,default=str)+"\n",encoding="utf8")
    temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        while data:=stream.read(1024*1024):
            h.update(data)
    return h.hexdigest()


def create_dataset(api, stage, record):
    metadata=json.loads((stage/"dataset-metadata.json").read_text(encoding="utf8"))
    capsule=json.loads((stage/"CAPSULE.json").read_text(encoding="utf8"))
    payload=stage/"payload.bin"
    if (payload.stat().st_size,sha(payload))!=(capsule["payload_bytes"],capsule["payload_sha256"]):
        raise ValueError("Frozen input payload drift")
    intent=record/"DATASET_CREATE_INTENT.json"
    if intent.exists():
        raise ValueError("Reconcile original dataset create before retrying")
    try:
        existing=api.dataset_status(metadata["id"],format="json(status,current_version_number)")
    except Exception as error:
        response=getattr(error,"response",None)
        code=getattr(response,"status_code",None)
        if code==403:
            matches=api.dataset_list(search=metadata["id"].split("/")[1],mine=True)
            if any(getattr(item,"ref",None)==metadata["id"] for item in matches):
                raise ValueError("Own dataset exists but status is unavailable") from None
            atomic_json(record/"DATASET_ABSENCE_CHECK.json",
                        {"ref":metadata["id"],"status_probe":403,"own_exact_candidates":[]})
        elif code!=404:
            raise RuntimeError("Dataset preflight failed: "+type(error).__name__) from None
    else:
        atomic_json(record/"DATASET_EXISTING.json",{"ref":metadata["id"],"status":existing})
        raise ValueError("Dataset already exists; reconcile its exact source")
    record.mkdir(parents=True,exist_ok=True)
    value={"ref":metadata["id"],"expected_version":1,"private":True,
           "payload_sha256":capsule["payload_sha256"],"payload_bytes":capsule["payload_bytes"],
           "input_manifest_sha256":capsule["manifest_sha256"],"automatic_retry":False,
           "observed_utc":datetime.now(timezone.utc).isoformat()}
    with intent.open("x",encoding="utf8") as stream:
        json.dump(value,stream,indent=2)
    result=subprocess.run([sys.executable,"-B","-m","kaggle","datasets","create","-p",".",
                            "-q","-t","-r","skip"],capture_output=True,text=True,encoding="utf8",errors="replace",cwd=stage.resolve())
    confirmed=result.returncode==0 and "Your private Dataset is being created" in result.stdout
    value.update(exit_code=result.returncode,confirmation=confirmed,
                 status="CREATE_CONFIRMED" if confirmed else "RECONCILE_ORIGINAL_REF")
    atomic_json(record/"DATASET_CREATE_RESPONSE.json",value)
    print(json.dumps(value),flush=True)
    if not confirmed:
        raise RuntimeError("Dataset create requires exact-ref reconciliation")


def save_kernel(api, stage, record, no_run):
    intent=json.loads((record/"SUBMISSION_INTENT.json").read_text(encoding="utf8"))
    metadata=json.loads((stage/"kernel-metadata.json").read_text(encoding="utf8"))
    if metadata["id"]!=intent["kernel"] or sha(stage/metadata["code_file"])!=intent["notebook_sha256"]:
        raise ValueError("Save source differs from frozen submission intent")
    if not metadata.get("is_private"):
        raise ValueError("Private notebook required")
    if (not no_run and intent.get("publisher_admission_status")!="QUALIFIED_AND_AUTHORIZED"):
        raise ValueError("Qualified publisher route required before scientific admission")
    attempt=record/"KERNEL_SAVE_ATTEMPT.json"
    if attempt.exists():
        raise ValueError("Reconcile original Save before retrying")
    candidates=api.kernels_list(mine=True,search=metadata["id"].split("/")[1])
    if any(getattr(item,"ref",None)==metadata["id"] for item in candidates):
        raise ValueError("Original kernel ref already exists; reconcile before Save")
    atomic_json(record/"KERNEL_ABSENCE_CHECK.json",
                {"ref":metadata["id"],"own_exact_candidates":[],
                 "utc":datetime.now(timezone.utc).isoformat()})
    with attempt.open("x",encoding="utf8") as stream:
        json.dump({"operation":"QUICK_SAVE" if no_run else "SAVE_AND_RUN_ALL",
                   "kernel":metadata["id"],"source_sha256":intent["notebook_sha256"],
                   "automatic_retry":False,"utc":datetime.now(timezone.utc).isoformat()},stream,indent=2)
    try:
        result=api.kernels_push(str(stage.resolve()),timeout=str(intent["provider_max_seconds"]),
                               acc="NvidiaTeslaT4" if metadata.get("enable_gpu") else None,no_run=no_run)
        value=type(result).to_dict(result)
        atomic_json(record/"KERNEL_SAVE_RESPONSE.json",value)
        print(json.dumps(value),flush=True)
    except Exception as error:
        atomic_json(record/"KERNEL_SAVE_UNCERTAIN.json",{"error_type":type(error).__name__,
                                                       "automatic_retry":False})
        raise RuntimeError("Save response uncertain; reconcile original ref") from None


def observe(api, ref, version, out):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiGetKernelSessionStatusRequest
    owner,slug=ref.split("/")
    with api.build_kaggle_client() as client:
        get=ApiGetKernelRequest()
        get.user_name,get.kernel_slug,get.version_label=owner,slug,f"v{version}"
        saved=client.kernels.kernels_api_client.get_kernel(get)
        request=ApiGetKernelSessionStatusRequest()
        request.user_name,request.kernel_slug,request.version_label=owner,slug,f"v{version}"
        status=client.kernels.kernels_api_client.get_kernel_session_status(request)
    result={"ref":ref,"version":version,"kernel_id":saved.metadata.id,"status":status.status.name,
            "is_private":saved.metadata.is_private,"utc":datetime.now(timezone.utc).isoformat()}
    atomic_json(out,result)
    with out.with_suffix(".history.jsonl").open("a",encoding="utf8") as stream:
        stream.write(json.dumps(result)+"\n")
    print(json.dumps(result),flush=True)


def read_output(client, ref, version, name, cap):
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    owner,slug=ref.split("/")
    request=ApiDownloadKernelOutputRequest()
    request.owner_slug,request.kernel_slug,request.version_number=owner,slug,version
    request.file_path=name
    data=bytearray()
    with client.kernels.kernels_api_client.download_kernel_output(request) as response:
        response.raise_for_status()
        if int(response.headers.get("Content-Length","0"))>cap:
            raise ValueError("Saved file exceeds byte limit")
        for block in response.iter_content(16384):
            data.extend(block)
            if len(data)>cap:
                raise ValueError("Saved stream exceeds byte limit")
    return bytes(data)


def persist_output(target, data):
    if target.exists():
        if target.read_bytes()!=data:
            raise ValueError("Existing local output differs; preserve it for reconciliation")
        return
    temp=target.with_suffix(target.suffix+".part")
    temp.write_bytes(data)
    if temp.read_bytes()!=data:
        raise ValueError("Local byte readback failed")
    temp.rename(target)


def original_run(record):
    intent=json.loads((record/"SUBMISSION_INTENT.json").read_text(encoding="utf8"))
    original=json.loads((record/"KERNEL_SAVE_RESPONSE.json").read_text(encoding="utf8"))
    observation=json.loads((record/"OBSERVATION-v1.json").read_text(encoding="utf8"))
    ref,version=original["ref"].removeprefix("/code/"),original["versionNumber"]
    if (observation["ref"],observation["version"],observation["kernel_id"]) != (
            ref,version,original["kernelId"]) or not observation["is_private"]:
        raise ValueError("Original private Kaggle version identity differs")
    return intent, original, observation


def collect(api, record, out, input_manifest):
    """Read only the compact files and one selected checkpoint from the original run."""
    intent,original,observation=original_run(record)
    if observation["status"]!="COMPLETE":
        raise ValueError("Original Kaggle version has not completed")
    if sha(input_manifest)!=intent["input_manifest_sha256"]:
        raise ValueError("Frozen input manifest identity differs")
    frozen=json.loads(input_manifest.read_text(encoding="utf8"))
    file_hashes={row["path"]:row["sha256"] for row in frozen["files"]}
    trained_sources={name:file_hashes[name] for name in (
        "train.py","roadnet.py","qa.py","sim.py","configs/field_layout.json")}
    ref,version=observation["ref"],observation["version"]
    run_id=intent["run_id"]
    out.mkdir(parents=True,exist_ok=True)
    limits={"-RESULT.json":64*1024,"-METRICS.json":512*1024,
            "-HISTORY.json":256*1024,"-BEST.pt":2*1024*1024,
            "/ENVIRONMENT.json":32*1024}
    outputs={}
    with api.build_kaggle_client() as client:
        for suffix, cap in limits.items():
            name=run_id+suffix
            outputs[name]=read_output(client,ref,version,name,cap)
    saved=json.loads(outputs[run_id+"-RESULT.json"])
    metrics=json.loads(outputs[run_id+"-METRICS.json"])
    history=json.loads(outputs[run_id+"-HISTORY.json"])
    env=json.loads(outputs[run_id+"/ENVIRONMENT.json"])
    best_sha=hashlib.sha256(outputs[run_id+"-BEST.pt"]).hexdigest()
    config=metrics["config"]
    if (saved["run_id"]!=run_id or saved["input_manifest_sha256"]!=intent["input_manifest_sha256"]
            or saved["compute_complete"] is not True or config["run_id"]!=run_id
            or config["device"]!="cuda" or not env["gpu"] or env["worker_sha256"]!=intent["worker_sha256"]
            or metrics["best_sha256"]!=best_sha or config["counts"]!={"train":4000,"val":500,"test":500}
            or config["manifest_sha256"]!=frozen["dataset_manifest_sha256"]
            or config["source_sha256"]!=trained_sources
            or any(config[key]!=frozen[wanted] for key,wanted in (
                ("epochs","epochs_max"),("seed","seed"),("batch","batch"),("patience","patience")))
            or saved["selected_epoch"]!=metrics["best_epoch"]
            or saved["optimizer_steps"]!=metrics["optimizer_steps"]
            or history[-1]["optimizer_steps"]!=metrics["optimizer_steps"]
            or metrics["actual_epochs"]!=len(history)
            or metrics["optimizer_steps"]!=len(history)*250
            or not 1<=metrics["actual_epochs"]<=frozen["epochs_max"]
            or [(r["epoch"],r["optimizer_steps"]) for r in history]!=[(i,i*250) for i in range(1,len(history)+1)]
            or max(history,key=lambda row:row["val_score"])["epoch"]!=metrics["best_epoch"]
            or metrics["test"]["frames"]!=500 or metrics["validation"]["frames"]!=500):
        raise ValueError("Saved compute files disagree with the frozen Full contract")
    rows=[]
    for name, data in outputs.items():
        target=out/Path(name).name
        persist_output(target,data)
        rows.append({"provider_path":name,"local_name":target.name,"bytes":len(data),
                     "sha256":hashlib.sha256(data).hexdigest()})
    result={"status":"VERIFIED_KAGGLE_GPU_COMPUTE","kernel_id":original["kernelId"],
            "ref":ref,"version":version,"run_id":run_id,"gpu":env["gpu"],"files":rows,
            "completed_epochs":len(history),"optimizer_steps":metrics["optimizer_steps"],
            "best_epoch":metrics["best_epoch"],"best_sha256":best_sha}
    atomic_json(out/"COLLECTION.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="files"}),flush=True)


def diagnose(api, record, out):
    """Collect only small stage witnesses if this exact version is terminal and unsuccessful."""
    intent,_,observation=original_run(record)
    if observation["status"] in {"QUEUED","RUNNING","PENDING","INITIALIZING","COMPLETE"}:
        raise ValueError("Expected terminal unsuccessful source for diagnostics")
    out.mkdir(parents=True,exist_ok=True)
    run_id=intent["run_id"]
    names=(run_id+"-RESULT.json",run_id+"/STATE.json",run_id+"/ENVIRONMENT.json",
           run_id+"/TRAIN.log",run_id+"/training/history.json")
    files=[]
    with api.build_kaggle_client() as client:
        for name in names:
            try:
                data=read_output(client,observation["ref"],observation["version"],name,256*1024)
            except Exception as error:
                files.append({"path":name,"available":False,"error_type":type(error).__name__,
                              "http_status":getattr(getattr(error,"response",None),"status_code",None)})
                continue
            target=out/Path(name).name
            persist_output(target,data)
            files.append({"path":name,"available":True,"bytes":len(data),"sha256":sha(target)})
    value={"source":observation,"run_id":run_id,"files":files,"automatic_replay":False}
    atomic_json(out/"DIAGNOSTICS.json",value)
    print(json.dumps(value),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="action",required=True)
    d=sub.add_parser("dataset")
    d.add_argument("stage",type=Path)
    d.add_argument("record",type=Path)
    s=sub.add_parser("save")
    s.add_argument("stage",type=Path)
    s.add_argument("record",type=Path)
    s.add_argument("--no-run",action="store_true")
    o=sub.add_parser("observe")
    o.add_argument("ref")
    o.add_argument("version",type=int)
    o.add_argument("out",type=Path)
    c=sub.add_parser("collect")
    c.add_argument("record",type=Path)
    c.add_argument("out",type=Path)
    c.add_argument("--input-manifest",type=Path,required=True)
    failure=sub.add_parser("diagnose")
    failure.add_argument("record",type=Path)
    failure.add_argument("out",type=Path)
    a=p.parse_args()
    api=KaggleApi()
    api.authenticate()
    try:
        if a.action=="dataset":
            create_dataset(api,a.stage,a.record)
        elif a.action=="save":
            save_kernel(api,a.stage,a.record,a.no_run)
        elif a.action=="observe":
            observe(api,a.ref,a.version,a.out)
        elif a.action=="collect":
            collect(api,a.record,a.out,a.input_manifest)
        else:
            diagnose(api,a.record,a.out)
    except Exception as error:
        print(json.dumps({"status":"STOPPED_RECONCILE","error_type":type(error).__name__}),flush=True)
        raise SystemExit(1) from None
