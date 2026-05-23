# Shared helper: load deploy/server.env and expose ssh/scp helpers.
# Dot-sourced by upload.ps1 / fetch.ps1 / remote.ps1.
$ErrorActionPreference = 'Stop'

$envFile = Join-Path $PSScriptRoot 'server.env'
if (-not (Test-Path $envFile)) {
    throw "Missing $envFile. Copy deploy/server.env.example to deploy/server.env and fill it in."
}

$cfg = @{}
foreach ($line in Get-Content $envFile) {
    $t = $line.Trim()
    if ($t -and -not $t.StartsWith('#') -and $t.Contains('=')) {
        $k, $v = $t.Split('=', 2)
        # strip inline comments and surrounding whitespace
        $v = ($v -split '\s+#', 2)[0].Trim()
        $cfg[$k.Trim()] = $v
    }
}

$script:SshHost   = $cfg['SSH_HOST']
$script:SshUser   = $cfg['SSH_USER']
$script:SshPort   = if ($cfg['SSH_PORT']) { $cfg['SSH_PORT'] } else { '22' }
$script:SshKey    = $cfg['SSH_KEY']
$script:RemoteDir = if ($cfg['REMOTE_DIR']) { $cfg['REMOTE_DIR'] } else { '~/hw3rl' }
$script:RemoteVenv = if ($cfg['REMOTE_VENV']) { $cfg['REMOTE_VENV'] } else { 'hw3' }
$script:Target    = "$SshUser@$SshHost"

if (-not $SshHost -or -not $SshUser) {
    throw "SSH_HOST and SSH_USER must be set in $envFile."
}

function Get-SshArgs {
    $a = @('-p', $SshPort)
    if ($SshKey) { $a += @('-i', $SshKey) }
    return $a
}

function Get-ScpArgs {
    # scp uses capital -P for the port (ssh uses lowercase -p).
    $a = @('-P', $SshPort)
    if ($SshKey) { $a += @('-i', $SshKey) }
    return $a
}

function Invoke-Remote([string]$Command) {
    & ssh @(Get-SshArgs) $Target $Command
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed (exit $LASTEXITCODE): $Command" }
}
