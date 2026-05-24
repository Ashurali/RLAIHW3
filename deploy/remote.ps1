# Control training on the server over SSH. The heavy work runs DETACHED on the
# server (nohup), so you can close the SSH session / laptop and it keeps going.
#
#   pwsh deploy/remote.ps1 -Action setup                     # venv + install (detached)
#   pwsh deploy/remote.ps1 -Action smoke                     # quick env check (foreground)
#   pwsh deploy/remote.ps1 -Action train -Config P1 -Seed 0  # one run (detached)
#   pwsh deploy/remote.ps1 -Action queue                     # run the queue (detached)
#   pwsh deploy/remote.ps1 -Action status                    # running jobs + logs
#   pwsh deploy/remote.ps1 -Action tail  -Log P1_s0          # live-follow a log
#   pwsh deploy/remote.ps1 -Action stop  -Log P1_s0          # kill a job by pid file
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('setup', 'smoke', 'train', 'queue', 'status', 'tail', 'stop')]
    [string]$Action,
    [string]$Config,
    [int]$Seed = 0,
    [string]$Log
)
. "$PSScriptRoot\_common.ps1"

$prefix = "cd $RemoteDir && REMOTE_VENV='$RemoteVenv' PY_BIN='$RemotePyBin'"

switch ($Action) {
    'setup' {
        Invoke-Remote "$prefix bash deploy/remote_setup.sh"
    }
    'smoke' {
        # Foreground: short, and you want to see the output directly.
        Invoke-Remote "$prefix && source deploy/_activate.sh && activate_env && python smoke_test.py"
    }
    'train' {
        if (-not $Config) { throw "-Config is required (e.g. -Config P1)" }
        $name = [IO.Path]::GetFileNameWithoutExtension($Config)
        Invoke-Remote "$prefix bash deploy/remote_train.sh configs/$name.yaml $Seed"
    }
    'queue' {
        Invoke-Remote "$prefix bash deploy/remote_queue.sh"
    }
    'status' {
        Invoke-Remote "cd $RemoteDir && echo '== running python jobs =='; (pgrep -a -f 'python -u' || echo none); echo; echo '== recent logs =='; (ls -lt logs 2>/dev/null | head -n 30 || echo 'no logs yet')"
    }
    'tail' {
        if (-not $Log) { throw "-Log is required (e.g. -Log P1_s0, or -Log queue_<timestamp>)" }
        Write-Host "Tailing $RemoteDir/logs/$Log.log  (Ctrl+C to stop following)"
        Invoke-Remote "tail -n 200 -f $RemoteDir/logs/$Log.log"
    }
    'stop' {
        if (-not $Log) { throw "-Log is required = pid file name (e.g. -Log P1_s0, -Log queue)" }
        Invoke-Remote "f=$RemoteDir/logs/$Log.pid; if [ -f `$f ]; then kill `$(cat `$f) && echo 'stopped pid '`$(cat `$f); else echo 'no pid file: '`$f; fi"
    }
}
