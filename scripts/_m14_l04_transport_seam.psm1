Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-L04RawCapture {
    param(
        [string]$Path,
        [string]$Stdout,
        [string]$Stderr
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $parent = Split-Path -Parent $fullPath
    if ($parent) {
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    $lf = [string][char]10
    $capture = "--- STDOUT BEGIN ---" + $lf + $Stdout + $lf + "--- STDOUT END ---" + $lf
    $capture += "--- STDERR BEGIN ---" + $lf + $Stderr + $lf + "--- STDERR END ---" + $lf
    $leaf = [System.IO.Path]::GetFileName($fullPath)
    $temporaryPath = Join-Path $parent ("." + $leaf + "." + [Guid]::NewGuid().ToString("N") + ".tmp")
    try {
        $encoding = [System.Text.UTF8Encoding]::new($false)
        $stream = [System.IO.FileStream]::new(
            $temporaryPath,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        try {
            $writer = [System.IO.StreamWriter]::new($stream, $encoding)
            try {
                $writer.Write($capture)
                $writer.Flush()
                $stream.Flush($true)
            } finally {
                $writer.Dispose()
            }
        } finally {
            $stream.Dispose()
        }
        if ([System.IO.File]::Exists($fullPath)) {
            [System.IO.File]::Replace($temporaryPath, $fullPath, $null, $true)
        } else {
            [System.IO.File]::Move($temporaryPath, $fullPath)
        }
        return $true
    } finally {
        if ([System.IO.File]::Exists($temporaryPath)) {
            try { [System.IO.File]::Delete($temporaryPath) } catch { }
        }
    }
}

function Invoke-L04TransportProcess {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$SshExecutable,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList,
        [Parameter(Mandatory = $true)]
        [byte[]]$BootstrapBytes,
        [Parameter(Mandatory = $true)]
        [string]$RawCapturePath
    )

    $state = @{
        stdout = ""
        stderr = ""
        transport_error = $null
        ssh_exit = $null
        process_started = $false
        stdin_closed = $false
        raw_capture_write_succeeded = $false
    }
    $setError = {
        param([System.Exception]$Exception)
        if ($state.transport_error -eq $null) {
            $state.transport_error = $Exception.GetType().Name
            $state.stderr += "L04_TRANSPORT_ERROR=$($state.transport_error)"
            $state.stderr += [string][char]10
        }
    }

    $process = $null
    $stdoutTask = $null
    $stderrTask = $null
    try {
        try {
            $psi = [System.Diagnostics.ProcessStartInfo]::new()
            $psi.FileName = $SshExecutable
            $psi.UseShellExecute = $false
            $psi.RedirectStandardInput = $true
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.CreateNoWindow = $true
            foreach ($argument in $ArgumentList) {
                [void]$psi.ArgumentList.Add($argument)
            }
            $process = [System.Diagnostics.Process]::new()
            $process.StartInfo = $psi
            if (-not $process.Start()) {
                throw [System.InvalidOperationException]::new("transport process did not start")
            }
            $state.process_started = $true
            $stdoutTask = $process.StandardOutput.ReadToEndAsync()
            $stderrTask = $process.StandardError.ReadToEndAsync()
        } catch {
            & $setError $_.Exception
        }

        if ($state.process_started -and $state.transport_error -eq $null) {
            try {
                $process.StandardInput.BaseStream.Write($BootstrapBytes, 0, $BootstrapBytes.Length)
                $process.StandardInput.Close()
                $state.stdin_closed = $true
            } catch {
                & $setError $_.Exception
                try { $process.StandardInput.Close(); $state.stdin_closed = $true } catch { & $setError $_.Exception }
            }
        }

        if ($state.process_started) {
            try {
                if (-not $process.WaitForExit(120000)) {
                    throw [System.TimeoutException]::new("transport process timeout")
                }
                $state.ssh_exit = $process.ExitCode
            } catch {
                & $setError $_.Exception
                try { if (-not $process.HasExited) { $process.Kill() } } catch { & $setError $_.Exception }
            }
        }
    } catch {
        & $setError $_.Exception
    } finally {
        if ($process -ne $null -and -not $state.stdin_closed) {
            try { $process.StandardInput.Close(); $state.stdin_closed = $true } catch { & $setError $_.Exception }
        }
        if ($stdoutTask -ne $null) {
            try { $state.stdout = $stdoutTask.GetAwaiter().GetResult() } catch { & $setError $_.Exception }
        }
        if ($stderrTask -ne $null) {
            try { $state.stderr += $stderrTask.GetAwaiter().GetResult() } catch { & $setError $_.Exception }
        }
        try {
            $state.raw_capture_write_succeeded = Write-L04RawCapture -Path $RawCapturePath -Stdout $state.stdout -Stderr $state.stderr
        } catch { & $setError $_.Exception }
        if ($process -ne $null) { $process.Dispose() }
    }

    $rawDigest = $null
    $rawWritten = $false
    if ($state.raw_capture_write_succeeded -and (Test-Path -LiteralPath $RawCapturePath -PathType Leaf)) {
        try {
            $rawDigest = ([Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData([System.IO.File]::ReadAllBytes($RawCapturePath)))).ToLowerInvariant()
            $rawWritten = $true
        } catch { & $setError $_.Exception }
    }
    [pscustomobject]@{
        ssh_exit = $state.ssh_exit
        raw_capture_sha256 = $rawDigest
        raw_capture_written_before_parse = $rawWritten
        raw_capture_write_succeeded = $state.raw_capture_write_succeeded
        transport_error = $state.transport_error
        exception_type = $state.transport_error
    }
}

Export-ModuleMember -Function Invoke-L04TransportProcess
