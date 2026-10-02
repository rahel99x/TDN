# Internal helpers. Run the public scripts with -File rather than dot-sourcing them.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-TdnRoot {
    $root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
    if (-not (Test-Path -LiteralPath (Join-Path $root 'pyproject.toml') -PathType Leaf)) {
        throw 'Cannot locate the extracted TDN project root.'
    }
    return $root
}

function Assert-TdnPath {
    param([Parameter(Mandatory=$true)][string]$Root,
          [Parameter(Mandatory=$true)][string]$Path)
    $full = [System.IO.Path]::GetFullPath($Path)
    $prefix = $Root.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if (($full -ne $Root) -and (-not $full.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Path escapes the project: $full"
    }
    # Refuse existing child junctions/symlinks before creating caches or a venv.
    $cursor = $full
    while (($cursor -ne $Root) -and ($cursor.Length -ge $Root.Length)) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Project child is a junction or symlink; preserve it and resolve manually: $cursor"
            }
        }
        $cursor = [System.IO.Path]::GetDirectoryName($cursor)
    }
    return $full
}

function Initialize-TdnEnvironment {
    param([Parameter(Mandatory=$true)][string]$Root, [int]$CudaDevice = 0)
    if ($env:SLURM_JOB_ID -or $env:SLURM_STEP_ID -or $env:SLURM_JOB_ACCOUNT) { throw 'Desktop execution cannot run inside a Slurm allocation.' }
    if ($env:CONDA_PREFIX) { throw 'Deactivate Conda and use standalone Python.' }
    if ($CudaDevice -lt 0) { throw 'CudaDevice must be nonnegative.' }
    foreach ($name in @('PIP_TARGET','PIP_PREFIX','PIP_USER','PIP_LOG','PIP_BUILD_TRACKER','PIP_EXTRA_INDEX_URL','PYTHONUSERBASE','PYTHONPATH')) {
        [System.Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
    $env:PIP_CONFIG_FILE = 'NUL'
    $env:TDN_EXECUTION_MODE = 'desktop'
    $env:TDN_PROJECT_ROOT = $Root
    $env:TDN_DESKTOP_CUDA_DEVICE = [string]$CudaDevice
    $locations = @{
        TMPDIR = '.runtime\tmp'; TMP = '.runtime\tmp'; TEMP = '.runtime\tmp'
        PIP_CACHE_DIR = '.runtime\cache\pip'; XDG_CACHE_HOME = '.runtime\cache'
        TORCH_HOME = '.runtime\cache\torch'; TORCHINDUCTOR_CACHE_DIR = '.runtime\cache\inductor'
        TRITON_CACHE_DIR = '.runtime\cache\triton'; TORCH_EXTENSIONS_DIR = '.runtime\cache\extensions'
        CUDA_CACHE_PATH = '.runtime\cache\cuda'; MPLCONFIGDIR = '.runtime\cache\matplotlib'
        PYTHONPYCACHEPREFIX = '.runtime\cache\pycache'
    }
    foreach ($entry in $locations.GetEnumerator()) {
        $target = Assert-TdnPath -Root $Root -Path (Join-Path $Root $entry.Value)
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        [System.Environment]::SetEnvironmentVariable($entry.Key, $target, 'Process')
    }
    $env:PYTHONNOUSERSITE = '1'
    $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
    $env:MPLBACKEND = 'Agg'
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
}

function Invoke-TdnChecked {
    param([Parameter(Mandatory=$true)][string]$Executable,
          [Parameter(Mandatory=$true)][string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Executable $($Arguments -join ' ')" }
}

function Get-TdnStandalonePython {
    param([string]$PythonExecutable = '')
    if ($env:CONDA_PREFIX) { throw 'Deactivate Conda and use standalone Python 3.11 or 3.12.' }
    $candidates = @()
    if ($PythonExecutable) {
        $candidates += ,@($PythonExecutable)
    } else {
        $launcher = Get-Command 'py' -ErrorAction SilentlyContinue
        if ($launcher) {
            $candidates += ,@($launcher.Source, '-3.11')
            $candidates += ,@($launcher.Source, '-3.12')
        }
        $python = Get-Command 'python' -ErrorAction SilentlyContinue
        if ($python) { $candidates += ,@($python.Source) }
    }
    $bootstrap = Join-Path (Get-TdnRoot) 'scripts\desktop.py'
    foreach ($candidate in $candidates) {
        $executable = $candidate[0]
        $arguments = @()
        if ($candidate.Count -gt 1) { $arguments += $candidate[1..($candidate.Count-1)] }
        $arguments += @($bootstrap, 'interpreter')
        try {
            $result = @(& $executable @arguments 2>$null)
            if (($LASTEXITCODE -eq 0) -and ($result.Count -gt 0)) {
                $path = ([string]$result[-1]).Trim()
                if (Test-Path -LiteralPath $path -PathType Leaf) { return $path }
            }
        } catch { continue }
    }
    throw 'Install standalone 64-bit Python 3.11 or 3.12 from python.org, or provide -PythonExecutable with its full path.'
}

function Get-TdnVenvPython {
    param([Parameter(Mandatory=$true)][string]$Root)
    $prefix = Assert-TdnPath -Root $Root -Path (Join-Path $Root '.venv')
    $python = Assert-TdnPath -Root $Root -Path (Join-Path $prefix 'Scripts\python.exe')
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Complete scripts\windows\Setup.ps1 first.' }
    if (-not (Test-Path -LiteralPath (Join-Path $prefix 'pyvenv.cfg') -PathType Leaf)) { throw 'The existing .venv is not a Python venv.' }
    return $python
}
