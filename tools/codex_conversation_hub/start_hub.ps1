param(
  [int]$Port = 8765,
  [switch]$NoBrowser
)

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$args = @('-m', 'tools.codex_conversation_hub', 'serve', '--port', $Port)
if (-not $NoBrowser) { $args += '--open' }
Push-Location $repoRoot
try {
  python @args
} finally {
  Pop-Location
}
