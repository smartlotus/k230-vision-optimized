"""Read-only host log evidence. Does not open serial or change VPN/settings."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re


def live_readiness(client,nonce):
    """Cached ready is diagnostic; a fresh REPL echo proves idle execution."""
    if not re.fullmatch(r'[A-Za-z0-9_]{8,64}',nonce):raise ValueError('Literal diagnostic nonce required')
    board=client.tool('canmv_board_info')
    if board.get('connected') is not True:raise RuntimeError('No connected board session')
    if client.tool('canmv_script_running').get('running') is not False:raise RuntimeError('Do not send REPL input to a busy board')
    client.tool('canmv_terminal_input',{'text':"print('"+nonce+"')\r"})
    import time
    for _ in range(10):
        text=client.tool('canmv_terminal_output',{'clear':False}).get('text','')
        if nonce in text.replace('\r\n','\n').split('\n'):
            return {'cached_ready':board.get('ready'),'fresh_repl_echo':True,
                    'interpreter_idle_before_probe':True,'readiness_basis':'FRESH_LITERAL_REPL_ECHO','nonce':nonce}
        time.sleep(.1)
    raise RuntimeError('No exact REPL echo; interpreter readiness remainsUNKNOWN')


def analyze(text):
    lines=text.splitlines()
    restarts=[(i,line) for i,line in enumerate(lines) if 'No frames received for startup; restarting preview' in line]
    disconnects=[(i,line) for i,line in enumerate(lines) if 'Disconnected (' in line or 'board disconnected source=' in line]
    errors=[line for line in lines if 'scriptRunning failed' in line or 'handshake' in line and ('failed' in line or 'rejected' in line)]
    return dict(preview_no_frame_restarts=len(restarts),disconnect_log_records=len(disconnects),
        preview_restarts_before_disconnect=any(0<j-i<=60 for i,_ in restarts for j,_ in disconnects),
        disconnects=[line for _,line in disconnects],protocol_errors=errors,
        inference='Concurrent preview/watchdog traffic is a plausible contributor; correlation does not establish a physical USB root cause')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--log',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    raw=args.log.read_bytes();report=analyze(raw.decode('utf-8',errors='replace'))
    report.update(created_utc=datetime.now(timezone.utc).isoformat(),log_sha256=hashlib.sha256(raw).hexdigest(),
                  source_log=str(args.log),serial_opened=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('preview_no_frame_restarts','disconnect_log_records','preview_restarts_before_disconnect')}))


if __name__=='__main__':main()
