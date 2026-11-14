"""Kaggle CLI wrapper. Reads an existing local token, never prints/stores it.

Pass a credential file path explicitly or configure TOPFACE_KAGGLE_TOKEN_FILE.
The token is inherited by the Kaggle subprocess environment, not its arguments.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--token-file",type=Path,
                        default=os.environ.get("TOPFACE_KAGGLE_TOKEN_FILE"))
    parser.add_argument("arguments",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    env=os.environ.copy()
    if args.token_file:
        token=Path(args.token_file).read_text(encoding="utf8").strip()
        if not token or any(char.isspace() for char in token):
            raise ValueError("Invalid Kaggle token file format")
        env["KAGGLE_API_TOKEN"]=token
    command=args.arguments
    if command and command[0]=="--":command=command[1:]
    if os.name=='nt' and command[:2] in (['datasets','create'],['datasets','version']):
        # The CLI's resumable upload state escapes the native path separator.
        # Normalize paths before the CLI uses one as a local state filename.
        for index,arg in enumerate(command[:-1]):
            if arg in ('-p','--path'):
                command[index+1]=str(Path(command[index+1]).resolve())
    result=subprocess.run([sys.executable,"-m","kaggle",*command],env=env)
    raise SystemExit(result.returncode)


if __name__=="__main__":main()
