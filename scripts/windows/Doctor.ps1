[CmdletBinding()]
param([string]$PythonExecutable = '', [int]$CudaDevice = 0)
. (Join-Path $PSScriptRoot 'Common.ps1')
try {
    $root = Get-TdnRoot
    Initialize-TdnEnvironment -Root $root -CudaDevice $CudaDevice
    $venvPython = Join-Path $root '.venv\Scripts\python.exe'
    if ((-not $PythonExecutable) -and (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        $python = Get-TdnVenvPython -Root $root
    } else {
        $python = Get-TdnStandalonePython -PythonExecutable $PythonExecutable
    }
    Invoke-TdnChecked -Executable $python -Arguments @((Join-Path $root 'scripts\desktop.py'), 'doctor', '--cuda-device', [string]$CudaDevice)
    exit 0
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
