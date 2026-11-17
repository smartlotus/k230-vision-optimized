# Kaggle execution and recovery experience

These lessons come from accepted ACT and K230 runs. Recheck the current account, SDK, execution mode, and provider response. Use the current project's handoff to identify the run, owner, destination, and any pause.

## Choose the qualified data path

Two cloud publication paths have worked in this project.

| Path | Compute and access | Publication and acceptance |
|---|---|---|
| Notebook to Drive | Private notebook with Internet and its attached runtime Secret | Notebook creates the archive in `/kaggle/working`, uploads it, and reads the complete stored bytes back |
| Offline compute with a cloud publisher | Private notebook mounts explicit cloud input versions and produces a saved output archive with Internet disabled | Authorized cloud publisher fetches that exact output, audits it, uploads to Drive, and reads every stored byte back |

The second path was used by native003 and the later CPU audits. Kaggle working storage holds temporary production data. Drive holds the accepted durable archive. The host collects bounded results, manifests, hashes, file IDs, and receipts. Moving a large archive through the host before uploading it defeats the user's cloud-first requirement.

Original historical inputs that exist only locally may need a one-time, explicitly scoped private input migration. Reuse an existing hash-matched cloud input when available. Distinguish input migration from the location where new process artifacts are produced.

## Submission identity and uncertain sends

Freeze the run ID, notebook source, parser and input hashes, explicit dataset/notebook versions, seeds and order, resource caps, result limits, and archive destination before execution. Record the provider ref, numeric kernel ID, and exact saved version alongside the source identity.

- Persist a submission intent before the Save request. Control write retries, including SDK decorators, so a lost response leads to reconciliation of the same submission.
- After an uncertain send, read the original ref and intended version, compare source and privacy settings, and locate the existing session. An intent alone leaves the outcome uncertain.
- A local lock file describes ownership. Verify the owning process before treating work as active or reclaiming ownership. Preserve locks and completed run records during diagnosis.
- `QUICK_SAVE` records a version. Execution requires a running mode such as the task's tested `SAVE_AND_RUN_ALL` path. Record the returned version and inspect its actual session.
- Editor draft sessions and saved versions have distinct identities. Collection binds the exact version and code cells. A newer editor version requires its own reconciliation.
- Technical retry may recover a transfer or an uncommitted unit within its frozen allowance. Scientific low performance follows the registered decision rule. Preserve charged attempts and the original budget.

The existing `research/ab_a_next/operate_cloud_audit.py` writes `FREEZE.json`, `SAVE_ATTEMPT.json`, and `SAVE_RESPONSE.json`, then verifies the exact saved source during collection. Reuse its task controls and writer ownership where applicable.

Validate notebook titles and slugs separately before Save, using the installed request schema and current provider constraints where exposed. Preserve a safe HTTP status and validation-field summary on failure. The observed B-line recovery used a shorter title and slug after a long-name request returned HTTP 400 and the owned inventory showed no saved version. That observation does not establish a universal title-length limit. A 403 from a private ref alone leaves permission and absence unresolved. Reconcile it against authenticated owned listings, an accessible account witness and any user-visible version before choosing a distinct replacement attempt.

If an unsent prepared notebook's latest-start timestamp expires, verify that it has no Save intent, response or active provider identity. Preserve its exact bytes before preparing a new start window. Compare the new notebook and preparation record and require any claimed deadline-only repair to change only that timestamp. Keep the scientific contract, inputs, seeds and active/lease limits bound. An existing Save uses its original version and cannot be renewed this way.

## Queues, GPU allowance, and CPU work

Observe the requested saved session before diagnosing a long queue. Record the observation time, provider state, accelerator type, current allowance and reservations, known active session IDs, and the account's actual capacity response. GPU time remaining and available concurrent capacity answer different questions. Queue duration alone cannot identify a provider policy or establish a failed submission.

Read durations with `total_seconds()` or an equivalent full-duration parser. Preserve the reset timestamp and its timezone. Recheck values before a new admission. Historical quota screenshots and previous runs provide dated observations.

This project completed CPU audits while GPU training was constrained. Use ready CPU-only parsing, provenance analysis, statistics, and package preparation within their own budgets. Choose independent cloud jobs according to actual provider capacity and ownership. A universal one-process rule, fixed weekly allowance, or fixed queue deadline would require current provider evidence.

Waiting needs the original session identity and bounded or adaptive read-only observation. A network timeout leaves that session uncertain. Re-read the same handle before deciding to cancel or replace it. A user-requested restart still requires reconciliation of the original operation and charged work.

## Distinguish missing output from a failed run

| Observation | Recovery for the original run |
|---|---|
| 404 for a checkpoint or output while the exact saved version is still running | Inspect that version's status and exposed artifacts. Use a verified checkpoint only if the run already publishes one. Otherwise wait for saved outputs and then verify their bytes and identity. |
| Runtime Secret lookup returns HTTP 400 | Test the intended execution path with a Boolean-only presence check. For direct upload, verify the editor binding and Internet for that notebook, then run one storage probe. If attachment remains unavailable, use an already authorized offline compute and cloud publisher route where available. |
| Drive says "invalid_grant" | Identify the failing credential path. If refresh authorization has expired or been revoked, request account-owner reauthorization through the original connection, revalidate identity and folder, then reconcile the existing archive and upload. |
| ConnectTimeout or TLS error during publication, or a failed receipt GET after dispatch | Scope the transport check to the failing host. Check whether the transfer was dispatched, completed, or accepted, then resume the exact operation at its last verified step. Retry a failed read of the existing job before considering another write. |

For a changed path, qualify it with a small separate probe. Preserve the saved version, charged work, and existing reservation when fixing credentials or transport.

## Runtime Secrets and authorized publication

### OAuth expiry recovery

In one inspected CLI2.2.5 SDK, `KaggleCredentials.access_token_has_expired()`
compared expiration with `now - 30 minutes`. This allowed a recently expired
cached token to pass local authentication. Private saved-version reads returned
403 and an authenticated owned listing returned401. These responses did not
prove the original job had stopped or disappeared.

Check the actual authentication source and only safe expiry metadata. For an
OAuth-cache route with a valid refresh credential, use the SDK's supported
`refresh_access_token()` method, then verify a bounded owned read and re-observe
the same saved run/version. Keep token values out of stdout, arguments and Git;
do not delete the cache, reauthorize another account or resubmit scientific work
merely to recover a read. A scoped wrapper can refresh before the recorded UTC
expiry with a small margin, without modifying the installed SDK. Do not assume
an OAuth-cache repair applies when an environment or token-file source is active.

Kaggle Secrets are configured inside the Notebook Editor under Add-ons → Secrets. Attach the existing label, such as `DOASTRA_GDRIVE_OAUTH_JSON`, to the intended notebook. `UserSecretsClient` resolves it inside an authorized runtime. A host CLI login grants access to that host's client context.

The installed notebook Save/Create request schemas inspected in this project exposed no Secret-binding fields. That result applies to the inspected SDK version. Avoid assuming inheritance across API Saves or copying secret-binding fields from unrelated Benchmark/container types. Test the intended mode with a bounded Boolean-only presence check, followed by the selected route's small storage probe. Reuse valid evidence when the relevant configuration is unchanged.

For a separate cloud publisher, verify that the user's existing authority covers both that compute/storage service and the credential destination. In the accepted private GitHub route, a short-lived Drive access token entered a temporary Secret. The host retained protected recovery state, and the Secret was removed after terminal publication. The user's Kaggle Secret remained available for its notebook use. Report cleanup accurately for each credential owner.

Keep refresh credentials, access tokens, resumable session URLs, and signed download URLs out of source, notebook payloads, Git, command arguments, receipts, and error logs. Pass required temporary credentials through the selected service's protected environment or Secret facility. Use safe error summaries because exception strings can include URLs. Base64 and log masking do not protect a credential placed in source.

Large immutable manifests belong in a byte-pinned publisher input or commit after content review. Bind a small temporary credential envelope to that manifest's SHA-256. Native003 recovered a 489-member archive with a 1,601-byte credential envelope. Those measured sizes describe that run. Check the destination service's current input limits when preparing a different workload.

## Diagnose Windows networking by endpoint

Check the real endpoint and transport layer involved in the failed request. The Kaggle website, `api.kaggle.com`, Google token endpoint, Drive endpoint, and GitHub may behave differently through one proxy. A local TCP connection to the proxy proves only the first connection. Follow with TLS and an authenticated, bounded read.

The project observed refused proxy ports, TLS EOF/reset failures, and WinError 10053. A later direct route for the Kaggle API passed authentication and submission while another proxy route failed. Use a command-local route, such as an appropriate `NO_PROXY`, only after testing that route. Keep global networking and unrelated user processes under their existing owner.

Python/OpenSSL and Windows native TLS can produce different outcomes. The existing `provider_snapshot.py` retries bounded reads, and `windows_read_transport.py` uses `curl.exe` for three allowlisted metadata RPCs. Its authorization headers go through stdin. Preserve that read-only boundary. The Save mutation has a separately reviewed single-send transport.

When a route fails, report the specific host and stage. Ask for user network intervention after supported, scoped alternatives have been checked. Recheck current connectivity when the user resumes. A former failure or approval denial needs current evidence before it determines the next action.

For the Windows CLI resumable Dataset upload path, use the dataset input directory as the subprocess working directory and pass `-p .`. In CLI 2.2.5, an absolute forward-slash path remained partially unescaped inside the upload state filename. Set a distinct temporary directory per independent upload to keep their resumable state files independent. On restart, preserve and reconcile a session only with its matching file identity. Reconcile the private Dataset listing and current operation before a new create.

In the verified route the Kaggle API worked direct while the Google blob upload host required an explicit command-local proxy. The presence of a Windows registry proxy and a `NO_PROXY` variable did not prove the Python process used that proxy. Test a small real service endpoint and then the bounded transfer. Keep the proxy override scoped to the Kaggle command and keep the API host on the verified direct route. Never emit resumable state tokens or signed URLs in diagnostics.

## Reproduce the cloud environment

A Kaggle Dataset created from an uploaded ZIP may mount individual expanded files even when the host upload archive was byte-pinned. Check the actual provider member listing first. Support the unpacked layout by binding the manifest bytes to a digest in the saved source and checking every expected mounted member's path, size and SHA-256 before execution. Preserve the archived input's separate host digest. For inputs with many internal members, a ZIP stream carried as an opaque .bin file kept the provider inventory to one file in a private test. Qualify this with a small exact-version download and a saved notebook mount check. Bind both the container and its inventory digest, then use a bounded ZIP reader independent of the filename. For an offline missing dependency, bind the wheel's name, byte count and official PyPI digest, match the saved container's Python ABI and architecture, and verify an actual operator call before starting the workload.


Use explicit private cloud input versions, original wheel/asset hashes, runtime-compatible Python and binary packages, and the task's pinned NumPy, MuJoCo, model and processor identities. The host's successful import does not establish the Kaggle environment. A previous notebook failed before model construction because MuJoCo was missing.

Assert `/kaggle/working` where required, then assemble the workload there. Keep complete process data and recovery state in cloud storage. Enforce per-file, expanded-input, archive, output and local collection limits appropriate to the task. Cloud RAM, cloud working disk, host RAM, and host disk have separate limits and measurements.

Derive snapshot and expanded-array bounds from the actual codec, declared shapes and accepted same-runtime artifacts before launching a replay. A full simulator checkpoint can be much larger than its image histories. Record the per-snapshot and total archive capacity independently. If a capacity check stops a frozen run, preserve its work ledger and evidence. Qualify a larger output bound in a distinct attempt with the same scientific operations and retain the first attempt's cost.

Isolated Python mode `-I` omits normal script/current-directory import paths. Explicitly add only the reviewed frozen module directory when sibling imports are required. The project's publisher bootstrap repaired this import path without repeating the scientific workload.

Preserve exact source bytes across Windows and Linux. Use appropriate `.gitattributes` for frozen inputs, hash bytes before submission and after cloud staging, and keep any derived lock-file repair as an auditable addition. A source mismatch requires understanding the changed bytes. Editing the expected hash alone would hide drift.

For Blender or native compilers, probe the actual render backend or helper binary in the target image. A GPU label and a PyTorch GPU smoke check provide separate evidence from a Cycles render. A local micro-render can validate geometry and labels ahead of cloud admission. When exact PNG bytes matter, check ancillary metadata as well as pixel values. Blender 5.3 wrote PNG date, render time, and source-path text even with the visible stamp disabled. For a new deterministic output profile, disable its individual date, render-time and filename flags and verify pixels, manifests, and complete file digests across two renders on the qualified backend. Preserve previously frozen archives and their original receipts. If a reviewed Linux wheel helper reports "Permission denied" after extraction, compare its executable mode to the original wheel member first. Repair the exact helper and rerun the smallest compile and Simulator fixture using the same source and inputs.

## Archive reconciliation and full acceptance

Treat these as separate recorded outcomes.

| Outcome | Required evidence |
|---|---|
| Compute finished | Exact saved source/version plus terminal compute records and charged counts |
| Durable archive accepted | Complete member manifest, configured destination, exact object ID/name, byte count, and independent SHA-256 of the entire stored byte stream |
| Scientific conclusion | Relevant original observations, full planned/attempted/valid/missing/UNKNOWN denominators, and the registered analysis |

After an uncertain upload, use the recorded reserved Drive file ID and resumable session when available. If neither was recorded, inspect the publication intent and query the target folder for the immutable run name and archive digest. Verify each candidate's identity and complete bytes before accepting it. Reconcile ambiguous matches before another write. If the folder has no matching committed or reserved object, use the original dispatch and job records to confirm the previous transfer has stopped or was never dispatched before recording a replacement reservation and starting one upload of the same frozen archive. An uncertain transfer status calls for further read-only recovery of that job. Query confirmed bytes and destination metadata. Refresh an expired access token or signed source URL against the same frozen archive identity. Reconcile any completed object before creating another session or object. Recovery of transport leaves completed science closed.

Audit archive/member hashes and the actual task records in the publisher before acceptance. An internal manifest may intentionally omit its own row because it predates itself. Pin that manifest's bytes externally and check the remaining member inventory exactly. Require safe member names and bounded decompression when consuming archives or compressed receipts.

A provider completion badge, successful upload response, metadata checksum, or partial range response leaves full byte verification outstanding. Match destination parents, privacy, file identity, size, and the computed readback digest. Persist compact acceptance atomically. Subsequent status commands should return the stored validated acceptance instead of publishing it again.

Use a bounded artifact or a validated compact log encoding when a receipt is too large for reliable log collection. Native003 used a compressed receipt with bounded decoding. Verify identity and hashes after decoding. Preserve the failed transfer attempt and append the successful recovery record with its actual publisher commit.

## Pause, closeout, and research focus

Local STOP guards block new admissions. A live cooperative cloud pause additionally needs request delivery, acknowledgement, a committed checkpoint, clean worker exit, and a verified resume comparison. Provider cancellation is an interruption. Qualify its checkpoint behavior before describing it as lossless pause.

Native003 verified injected cloud-local STOP/restart comparisons. The manual Windows-to-cloud request path remains unqualified in its acceptance. Read the current evidence before extending that claim.

On a user pause, complete the authorized current stage and safe archival boundary, observe relevant sessions terminal or verify their checkpointed pause, preserve the existing scoped STOP, update the bounded handoff, and save an explicit resume entrypoint. Skill or documentation maintenance keeps that pause effective. Resume requires the user's instruction.

Use accepted checks within their verified source/configuration scope. Each additional audit should resolve a concrete obstacle to the next scientific decision and have a finite completion condition. Where historical metadata is incomplete, seek a sufficient original-state witness for the required independence claim. Full recovery of unrelated historical code is useful only when that claim depends on it. Preserve unresolved evidence and outcome-independent selection.

## Evidence and entrypoints

Resolve the A task checkout from `.ops/PROJECT.md`. The paths below are relative to that checkout, and the accepted files are also present in its private backup history. Current task ownership and run status still come from the current handoff and original provider records.

| Evidence | Verified scope |
|---|---|
| `reports/ab_a_next/native_gpu_003/ACCEPTANCE.json` and `CLOUD_RECOVERY_002.json` | 142,050,623 archive bytes, 489 members, full Drive readback, original upload-session recovery, temporary envelope and credential cleanup |
| `reports/ab_a_next/legacy_catalog_cloud_001/VERIFICATION.json` | Kaggle CPU 144970235 v1, 65.85 active seconds, 854,242 archive bytes, 3,953 compact compute bytes collected locally |
| `reports/ab_a_next/legacy_scalar_cloud_001/VERIFICATION.json` | Kaggle CPU 144972160 v1, original scalar comparison, explicit missing records and full archive acceptance |
| `reports/ab_a_next/PROVIDER_CONTROL_INTERFACE_001.md` and its source witnesses | Installed SDK Secret/cancellation surface and limits of the measured control route |
| `reports/ab_a_next/PAUSE_CLOSEOUT_20261114.json` | Four exact original versions terminal and explicit A-line user pause |
| `research/ab_a_next/operate_cloud_audit.py`, `operate_a1_archive.py`, and `auxiliary_cpu_ci_transfer.py` | Existing single-writer compute, exact-version collection and cloud publication profiles |
| `research/ab_a_next/native_transfer_envelope.py` and `native_publisher_bootstrap.py` | Static manifest binding, bounded receipt encoding and isolated import repair |
| `research/ab_a_next/continuation_control.py` | Existing A-line admission pause and explicit resume interface |

Keep provider IDs, ports, archive sizes and measured counts in their original receipts. New tasks use their own frozen identities and limits.
