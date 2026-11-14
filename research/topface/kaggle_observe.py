"""Bounded read-only observation of one exact saved Kaggle version."""
import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
from cloud_store import atomic_json


def observe(ref, version, token_file, output=None):
    if token_file:
        os.environ["KAGGLE_API_TOKEN"]=Path(token_file).read_text(encoding="utf8").strip()
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiGetKernelSessionStatusRequest
    user,slug=ref.split("/")
    request=ApiGetKernelRequest();request.user_name=user;request.kernel_slug=slug
    with api.build_kaggle_client() as client:
        metadata=client.kernels.kernels_api_client.get_kernel(request).metadata
        status_request=ApiGetKernelSessionStatusRequest()
        status_request.user_name=user;status_request.kernel_slug=slug;status_request.version_label=f"v{version}"
        status=client.kernels.kernels_api_client.get_kernel_session_status(status_request)
    raw=api.kernels_logs(f"{ref}/{version}")
    try:
        events=json.loads(raw) if raw else []
        lines="".join(event.get("data","") for event in events).splitlines()
    except (json.JSONDecodeError,TypeError):lines=raw.splitlines()
    result={"observed_utc":datetime.now(timezone.utc).isoformat(),"ref":ref,
        "kernel_id":metadata.id,"version":version,"current_version":metadata.current_version_number,
        "status":status.status.name,"log_tail":lines[-24:]}
    if output:
        output=Path(output)
        atomic_json(output,result)
        output.with_suffix(".log.json").write_text(raw,encoding="utf8")
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--ref",required=True)
    parser.add_argument("--version",required=True,type=int)
    parser.add_argument("--token-file",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    observe(args.ref,args.version,args.token_file,args.output)
