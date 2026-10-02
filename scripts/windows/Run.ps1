[CmdletBinding()]
param(
    [ValidateSet('cpu','cuda')][string]$Device = 'cpu',
    [string]$Config = 'configs/desktop-smoke.yaml',
    [ValidateRange(1,1000)][int]$MaxSteps = 12,
    [string]$RunId = '',
    [int]$CudaDevice = 0,
    [switch]$DryRun,
    [switch]$Resume
)
. (Join-Path $PSScriptRoot 'Common.ps1')
try {
    $root = Get-TdnRoot
    Initialize-TdnEnvironment -Root $root -CudaDevice $CudaDevice
    $python = Get-TdnVenvPython -Root $root
    $arguments = @((Join-Path $root 'scripts\desktop.py'), 'run', '--config', $Config,
                   '--device', $Device, '--cuda-device', [string]$CudaDevice, '--max-steps', [string]$MaxSteps)
    if ($RunId) { $arguments += @('--run-id', $RunId) }
    if ($DryRun) { $arguments += '--dry-run' }
    if ($Resume) { $arguments += '--resume' }
    & $python @arguments
    # Preserve exit 75 for a resumable training pause and other actual child status.
    exit $LASTEXITCODE
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
