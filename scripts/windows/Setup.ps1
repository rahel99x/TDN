[CmdletBinding()]
param(
    [string]$PythonExecutable = '',
    [string]$TorchVersion = '2.10.0',
    [ValidateSet('https://download.pytorch.org/whl/cpu', 'https://download.pytorch.org/whl/cu126', 'https://download.pytorch.org/whl/cu128', 'https://download.pytorch.org/whl/cu130')]
    [string]$TorchIndex = 'https://download.pytorch.org/whl/cpu'
)
. (Join-Path $PSScriptRoot 'Common.ps1')
try {
    $root = Get-TdnRoot
    Initialize-TdnEnvironment -Root $root
    if ($TorchVersion -notmatch '^\d+\.\d+\.\d+([+][A-Za-z0-9.]+)?$') { throw 'TorchVersion must be an exact release.' }
    $basePython = Get-TdnStandalonePython -PythonExecutable $PythonExecutable
    if ($TorchIndex -ne 'https://download.pytorch.org/whl/cpu') {
        $smi = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
        if (-not $smi) { throw 'CUDA setup needs nvidia-smi from the installed NVIDIA driver. Run Doctor.ps1 first; CPU mode remains available.' }
        $drivers = @(& $smi.Source '--query-gpu=driver_version' '--format=csv,noheader,nounits')
        if (($LASTEXITCODE -ne 0) -or ($drivers.Count -eq 0)) { throw 'Cannot observe the NVIDIA driver; use Doctor.ps1 or CPU mode.' }
        $minimum = switch ($TorchIndex) {
            'https://download.pytorch.org/whl/cu126' { [version]'560.76' }
            'https://download.pytorch.org/whl/cu128' { [version]'570.65' }
            'https://download.pytorch.org/whl/cu130' { [version]'580.88' }
        }
        foreach ($driver in $drivers) {
            $observed = [version](([string]$driver).Trim())
            if ($observed -lt $minimum) {
                throw "CUDA wheel requires Windows driver >= $minimum; observed $observed. Use CPU mode or deliberately update your NVIDIA driver yourself."
            }
        }
    }
    $prefix = Assert-TdnPath -Root $root -Path (Join-Path $root '.venv')
    if (Test-Path -LiteralPath $prefix) {
        $python = Get-TdnVenvPython -Root $root
        Invoke-TdnChecked -Executable $python -Arguments @((Join-Path $root 'scripts\desktop.py'), 'verify-venv', '--base-python', $basePython)
    } else {
        Invoke-TdnChecked -Executable $basePython -Arguments @('-m', 'venv', $prefix)
        $python = Get-TdnVenvPython -Root $root
    }
    Invoke-TdnChecked -Executable $python -Arguments @('-m', 'pip', 'install', 'setuptools==80.9.0', 'wheel==0.45.1')
    # Force replacement when selecting CPU versus CUDA; a public-version pin
    # alone can otherwise retain an installed build from another wheel index.
    Invoke-TdnChecked -Executable $python -Arguments @('-m', 'pip', 'install', '--force-reinstall', "torch==$TorchVersion", '--index-url', $TorchIndex)
    Invoke-TdnChecked -Executable $python -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $root 'requirements.txt'))
    Invoke-TdnChecked -Executable $python -Arguments @('-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', $root)
    Invoke-TdnChecked -Executable $python -Arguments @('-m', 'pip', 'check')
    $freeze = Assert-TdnPath -Root $root -Path (Join-Path $root '.runtime\windows-environment-freeze.txt')
    $freezeLines = @(& $python '-m' 'pip' 'freeze')
    if ($LASTEXITCODE -ne 0) { throw 'pip freeze failed.' }
    $freezeLines | Set-Content -LiteralPath $freeze -Encoding UTF8
    Invoke-TdnChecked -Executable $python -Arguments @((Join-Path $root 'scripts\desktop.py'), 'environment', '--torch-index', $TorchIndex)
    if ($TorchIndex -ne 'https://download.pytorch.org/whl/cpu') {
        Invoke-TdnChecked -Executable $python -Arguments @((Join-Path $root 'scripts\desktop.py'), 'gpu-check')
    }
    Write-Host "Installed project venv: $prefix"
    Write-Host 'Next: Run.ps1 -Device cpu, or -Device cuda for the explicitly selected CUDA build.'
    exit 0
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
