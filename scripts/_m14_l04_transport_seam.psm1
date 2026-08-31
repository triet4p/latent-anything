Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:L04KillGraceSeconds = 30

function Get-L04Sha256Hex {
    param([byte[]]$Bytes)

    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha.ComputeHash($Bytes)
    } finally {
        $sha.Dispose()
    }
    return ([BitConverter]::ToString($digest) -replace "-", "").ToLowerInvariant()
}

function ConvertTo-L04WindowsArgument {
    param([string]$Value)

    if ($null -eq $Value -or $Value.Length -eq 0) { return '""' }
    if ($Value -notmatch '[\s"]') { return $Value }

    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    for ($index = 0; $index -lt $Value.Length; $index++) {
        $character = $Value[$index]
        if ($character -eq [char]92) {
            $backslashes++
            continue
        }
        if ($character -eq [char]34) {
            if ($backslashes -gt 0) {
                [void]$builder.Append([char]92, ($backslashes * 2) + 1)
            } else {
                [void]$builder.Append([char]92)
            }
            [void]$builder.Append([char]34)
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append([char]92, $backslashes)
            $backslashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append([char]92, $backslashes * 2)
    }
    [void]$builder.Append([char]34)
    return $builder.ToString()
}

function Set-L04ProcessArguments {
    param(
        [Parameter(Mandatory = $true)] [System.Diagnostics.ProcessStartInfo]$ProcessStartInfo,
        [Parameter(Mandatory = $true)] [AllowEmptyCollection()] [AllowEmptyString()] [string[]]$ArgumentList
    )

    foreach ($argument in $ArgumentList) {
        if ($null -eq $argument) { throw "ArgumentList must not contain null" }
    }
    $argumentListProperty = $ProcessStartInfo.GetType().GetProperty("ArgumentList")
    if ($null -ne $argumentListProperty) {
        $argumentCollection = $argumentListProperty.GetValue($ProcessStartInfo, $null)
        foreach ($argument in $ArgumentList) {
            [void]$argumentCollection.Add($argument)
        }
        return "ArgumentList"
    }

    $quotedArguments = @()
    foreach ($argument in $ArgumentList) {
        $quotedArguments += ConvertTo-L04WindowsArgument $argument
    }
    $ProcessStartInfo.Arguments = [string]::Join(" ", [string[]]$quotedArguments)
    return "Arguments"
}

function Write-L04SingleFlightMetadata {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [byte[]]$Bytes
    )

    $directory = [System.IO.Path]::GetDirectoryName($Path)
    $temporaryPath = Join-Path $directory ("." + [Guid]::NewGuid().ToString("N") + ".tmp")
    $stream = $null
    try {
        $stream = [System.IO.FileStream]::new(
            $temporaryPath,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None,
            512,
            [System.IO.FileOptions]::WriteThrough
        )
        $stream.Write($Bytes, 0, $Bytes.Length)
        $stream.Flush($true)
        $stream.Dispose()
        $stream = $null
        if ([System.IO.File]::Exists($Path)) {
            $backupPath = $Path + "." + [Guid]::NewGuid().ToString("N") + ".bak"
            [System.IO.File]::Replace($temporaryPath, $Path, $backupPath, $true)
            try { [System.IO.File]::Delete($backupPath) } catch { }
        } else {
            [System.IO.File]::Move($temporaryPath, $Path)
        }
    } finally {
        if ($null -ne $stream) { try { $stream.Dispose() } catch { } }
        if ([System.IO.File]::Exists($temporaryPath)) {
            try { [System.IO.File]::Delete($temporaryPath) } catch { }
        }
    }
}

function New-L04GlobalMutexSecurity {
    <#
    Build the narrow ACL needed by a named synchronization object.  The
    constructor below is intentionally feature-detected by the caller: a
    runtime that cannot create this descriptor must fail closed rather than
    silently downgrading a host-wide guard to a session-local one.
    #>
    try {
        $security = [System.Security.AccessControl.MutexSecurity]::new()
        $rights = [System.Security.AccessControl.MutexRights]::Synchronize -bor
            [System.Security.AccessControl.MutexRights]::Modify
        $allow = [System.Security.AccessControl.AccessControlType]::Allow
        $currentSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
        if ($null -eq $currentSid) { throw "missing current identity" }
        $security.AddAccessRule(
            [System.Security.AccessControl.MutexAccessRule]::new($currentSid, $rights, $allow)
        )
        $systemSid = [System.Security.Principal.SecurityIdentifier]::new(
            [System.Security.Principal.WellKnownSidType]::LocalSystemSid, $null
        )
        $security.AddAccessRule(
            [System.Security.AccessControl.MutexAccessRule]::new($systemSid, $rights, $allow)
        )
        return $security
    } catch {
        throw "single_flight_global_mutex_security_unavailable"
    }
}

function Enter-L04SingleFlightLock {
    param(
        [Parameter(Mandatory = $true)] [string]$Key,
        [Parameter(Mandatory = $true)] [AllowEmptyCollection()] [AllowEmptyString()] [string[]]$ArgumentList
    )

    if ([string]::IsNullOrWhiteSpace($Key)) { throw "single_flight_key_invalid" }
    if ($null -eq $ArgumentList) { throw "single_flight_arguments_invalid" }
    foreach ($argument in $ArgumentList) {
        if ($null -eq $argument) { throw "single_flight_arguments_invalid" }
    }
    $root = Join-Path ([System.IO.Path]::GetTempPath()) "latent-anything-l04-single-flight"
    [System.IO.Directory]::CreateDirectory($root) | Out-Null
    $keyDigest = Get-L04Sha256Hex ([System.Text.UTF8Encoding]::new($false).GetBytes($Key))
    $mutexName = "Global\latent-anything-l04-" + $keyDigest
    $metadataPath = Join-Path $root ("l04-" + $keyDigest + ".json")
    $argumentDigest = Get-L04Sha256Hex (
        [System.Text.UTF8Encoding]::new($false).GetBytes([string]::Join([char]0, [string[]]$ArgumentList))
    )
    $ownerStart = [DateTime]::UtcNow.ToString("o", [Globalization.CultureInfo]::InvariantCulture)
    $metadata = [ordered]@{
        schema = "m14-l04-single-flight-v2"
        key_sha256 = $keyDigest
        owner_pid = [int]$PID
        owner_start_utc = $ownerStart
        argv_sha256 = $argumentDigest
    }
    $metadataBytes = [Text.UTF8Encoding]::new($false).GetBytes(($metadata | ConvertTo-Json -Compress))
    $mutex = $null
    $acquired = $false
    try {
        $createdNew = $false
        $mutexType = [System.Threading.Mutex]
        $boolRefType = [bool].MakeByRefType()
        $securityCtor = $null
        $optionsCtor = $null
        $legacyCtor = $null
        $namedWaitHandleOptionsType = $null
        foreach ($constructor in $mutexType.GetConstructors()) {
            $parameterTypes = @($constructor.GetParameters() | ForEach-Object { $_.ParameterType })
            foreach ($parameterType in $parameterTypes) {
                if ($parameterType.FullName -eq "System.Threading.NamedWaitHandleOptions") {
                    $namedWaitHandleOptionsType = $parameterType
                }
            }
            if ($parameterTypes.Count -eq 4 -and
                $parameterTypes[0] -eq [bool] -and $parameterTypes[1] -eq [string] -and
                $parameterTypes[2] -eq $boolRefType -and
                $parameterTypes[3] -eq [System.Security.AccessControl.MutexSecurity]) {
                $securityCtor = $constructor
            } elseif ($parameterTypes.Count -eq 3 -and
                $parameterTypes[0] -eq [bool] -and $parameterTypes[1] -eq [string] -and
                $null -ne $namedWaitHandleOptionsType -and
                $parameterTypes[2] -eq $namedWaitHandleOptionsType) {
                $optionsCtor = $constructor
            } elseif ($parameterTypes.Count -eq 3 -and
                $parameterTypes[0] -eq [bool] -and $parameterTypes[1] -eq [string] -and
                $parameterTypes[2] -eq $boolRefType) {
                $legacyCtor = $constructor
            }
        }
        try {
            if ($null -ne $securityCtor) {
                $security = New-L04GlobalMutexSecurity
                $mutex = [System.Threading.Mutex]::new($false, $mutexName, [ref]$createdNew, $security)
            } elseif ($null -ne $optionsCtor) {
                # Modern .NET removed the ACL-bearing constructor.  The OS
                # default for a Global synchronization object is retained;
                # never downgrade the namespace to Local/session scope.
                # Use the three-argument overload because PowerShell 5.1 and
                # 7 cannot bind the ref overload consistently.  Ownership is
                # established by WaitOne below, so createdNew is not needed.
                $waitHandleOptions = [Activator]::CreateInstance($namedWaitHandleOptionsType)
                # The default struct is CurrentSessionOnly/CurrentUserOnly;
                # clear both flags so the Global namespace is truly host-wide.
                $waitHandleOptions.CurrentSessionOnly = $false
                $waitHandleOptions.CurrentUserOnly = $false
                $constructorArguments = [object[]]@(
                    $false, $mutexName, $waitHandleOptions
                )
                $mutex = $optionsCtor.Invoke($constructorArguments)
            } elseif ($null -ne $legacyCtor) {
                $mutex = [System.Threading.Mutex]::new($false, $mutexName, [ref]$createdNew)
            } else {
                throw "unsupported named mutex constructor"
            }
        } catch {
            throw "single_flight_global_mutex_unavailable"
        }
        try {
            $acquired = $mutex.WaitOne(0)
        } catch [System.Threading.AbandonedMutexException] {
            # WaitOne has transferred ownership to this process despite the abandoned owner.
            $acquired = $true
        } catch {
            throw "single_flight_global_mutex_unavailable"
        }
        if (-not $acquired) {
            $mutex.Dispose()
            $mutex = $null
            throw "single_flight_busy"
        }
        try {
            Write-L04SingleFlightMetadata -Path $metadataPath -Bytes $metadataBytes
        } catch {
            try { $mutex.ReleaseMutex() } catch { }
            $mutex.Dispose()
            $mutex = $null
            throw "single_flight_metadata_write_failed"
        }
        return [pscustomobject]@{
            mutex = $mutex
            mutex_name = $mutexName
            metadata_path = $metadataPath
            key_sha256 = $keyDigest
            argv_sha256 = $argumentDigest
            owned = $true
        }
    } catch {
        if ($null -ne $mutex) {
            if ($acquired) { try { $mutex.ReleaseMutex() } catch { } }
            try { $mutex.Dispose() } catch { }
        }
        throw
    }
}

function Exit-L04SingleFlightLock {
    param([AllowNull()] [object]$Lock)

    if ($null -eq $Lock) { return }
    $mutex = $Lock.mutex
    if ($null -ne $mutex) {
        try {
            if ($Lock.owned -eq $true) { $mutex.ReleaseMutex() }
        } catch { }
        try { $mutex.Dispose() } catch { }
    }
}

function Observe-L04Task {
    param([System.Threading.Tasks.Task]$Task)

    if ($null -eq $Task) { return }
    if ($Task.IsFaulted) {
        try { $null = $Task.Exception } catch { }
    }
}

function Get-L04RemainingMilliseconds {
    param([long]$DeadlineTicks)

    $remainingTicks = $DeadlineTicks - [System.Diagnostics.Stopwatch]::GetTimestamp()
    if ($remainingTicks -le 0) { return 0 }
    $milliseconds = [math]::Ceiling(
        ([double]$remainingTicks * 1000.0) / [double][System.Diagnostics.Stopwatch]::Frequency
    )
    return [int][math]::Min($milliseconds, [int]::MaxValue)
}

function Wait-L04TaskUntilDeadline {
    param(
        [Parameter(Mandatory = $true)] [System.Threading.Tasks.Task]$Task,
        [Parameter(Mandatory = $true)] [long]$DeadlineTicks
    )

    if ($Task.IsCompleted) { return $true }
    $remainingMilliseconds = Get-L04RemainingMilliseconds $DeadlineTicks
    if ($remainingMilliseconds -le 0) { return $false }
    return $Task.Wait($remainingMilliseconds)
}

function Get-L04CompletedTaskValue {
    param([Parameter(Mandatory = $true)] [System.Threading.Tasks.Task]$Task)

    if (-not $Task.IsCompleted) { return $null }
    try {
        # The task is known complete before Result is read; this cannot block.
        return $Task.Result
    } catch {
        throw $_.Exception
    }
}

function Write-L04RawCapture {
    param(
        [string]$Path,
        [string]$Stdout,
        [string]$Stderr,
        [long]$DeadlineTicks
    )

    if ((Get-L04RemainingMilliseconds $DeadlineTicks) -le 0) {
        throw [System.TimeoutException]::new("raw capture deadline expired")
    }
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $parent = Split-Path -Parent $fullPath
    if ($parent) {
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    $lf = [string][char]10
    $capture = "--- STDOUT BEGIN ---" + $lf + $Stdout + $lf + "--- STDOUT END ---" + $lf
    $capture += "--- STDERR BEGIN ---" + $lf + $Stderr + $lf + "--- STDERR END ---" + $lf
    $captureBytes = [System.Text.UTF8Encoding]::new($false).GetBytes($capture)
    $leaf = [System.IO.Path]::GetFileName($fullPath)
    $temporaryPath = Join-Path $parent ("." + $leaf + "." + [Guid]::NewGuid().ToString("N") + ".tmp")
    try {
        $stream = [System.IO.FileStream]::new(
            $temporaryPath,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None,
            65536,
            $true
        )
        try {
            $stream.Write($captureBytes, 0, $captureBytes.Length)
            $stream.Flush($true)
        } finally {
            $stream.Dispose()
        }
        if ((Get-L04RemainingMilliseconds $DeadlineTicks) -le 0) {
            throw [System.TimeoutException]::new("raw capture publication deadline expired")
        }
        if ([System.IO.File]::Exists($fullPath)) {
            $backupPath = $temporaryPath + ".backup"
            try {
                [System.IO.File]::Replace($temporaryPath, $fullPath, $backupPath, $true)
            } finally {
                if ([System.IO.File]::Exists($backupPath)) {
                    [System.IO.File]::Delete($backupPath)
                }
            }
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
        [Parameter(Mandatory = $true)] [string]$SshExecutable,
        [Parameter(Mandatory = $true)] [AllowEmptyCollection()] [AllowEmptyString()] [string[]]$ArgumentList,
        [Parameter(Mandatory = $true)] [byte[]]$BootstrapBytes,
        [Parameter(Mandatory = $true)] [string]$RawCapturePath,
        [ValidateRange(1, 7200)] [int]$TimeoutSeconds = 3600,
        [string]$SingleFlightKey = "",
        [AllowNull()] [object]$SingleFlightLock = $null
    )

    if ($null -eq $ArgumentList) { throw "ArgumentList must not be null" }
    foreach ($argument in $ArgumentList) {
        if ($null -eq $argument) { throw "ArgumentList must not contain null" }
    }
    $deadlineTicks = [System.Diagnostics.Stopwatch]::GetTimestamp() + [long](
        $TimeoutSeconds * [System.Diagnostics.Stopwatch]::Frequency
    )
    $state = @{
        stdout = ""
        stderr = ""
        transport_error = $null
        transport_errors = [System.Collections.Generic.List[string]]::new()
        ssh_exit = $null
        process_started = $false
        stdin_closed = $false
        raw_capture_write_succeeded = $false
        deadline_exceeded = $false
        transport_termination_incomplete = $false
        cleanup_status = "not_required"
        raw_capture_path = $null
        raw_capture_sha256 = $null
        raw_capture_finalization_error = $null
    }
    $recordError = {
        param([string]$ErrorType)
        if ([string]::IsNullOrWhiteSpace($ErrorType)) { return }
        if (-not $state.transport_errors.Contains($ErrorType)) {
            [void]$state.transport_errors.Add($ErrorType)
        }
        if ($state.transport_error -eq $null) {
            $state.transport_error = $ErrorType
        }
        $state.stderr += "L04_TRANSPORT_ERROR=$ErrorType"
        $state.stderr += [string][char]10
    }
    $recordException = {
        param([System.Exception]$Exception)
        & $recordError $Exception.GetType().Name
    }
    $markDeadline = {
        $state.deadline_exceeded = $true
        & $recordError "TimeoutException"
    }
    $killAndDrainDeadline = {
        return [System.Diagnostics.Stopwatch]::GetTimestamp() + [long](
            $script:L04KillGraceSeconds * [System.Diagnostics.Stopwatch]::Frequency
        )
    }
    $publishRawCapture = {
        param([long]$CaptureDeadline, [string]$Phase)
        if ($state.raw_capture_write_succeeded) { return }
        try {
            $published = Write-L04RawCapture `
                -Path $RawCapturePath `
                -Stdout $state.stdout `
                -Stderr $state.stderr `
                -DeadlineTicks $CaptureDeadline
            if ($published) {
                $state.raw_capture_write_succeeded = $true
                $state.raw_capture_path = $RawCapturePath
                if (Test-Path -LiteralPath $RawCapturePath -PathType Leaf) {
                    $state.raw_capture_sha256 = Get-L04Sha256Hex (
                        [System.IO.File]::ReadAllBytes($RawCapturePath)
                    )
                }
            }
        } catch {
            $errorType = $_.Exception.GetType().Name
            if ($Phase -eq "finalization") {
                $state.raw_capture_finalization_error = $errorType
            }
            & $recordException $_.Exception
        }
    }

    $process = $null
    $stdoutTask = $null
    $stderrTask = $null
    $writeTask = $null
    $flushTask = $null
    $killDeadlineTicks = $null
    $ownedSingleFlightLock = $false
    if ($null -eq $SingleFlightLock -and -not [string]::IsNullOrWhiteSpace($SingleFlightKey)) {
        $SingleFlightLock = Enter-L04SingleFlightLock -Key $SingleFlightKey -ArgumentList $ArgumentList
        $ownedSingleFlightLock = $true
    }
    try {
        try {
            if ((Get-L04RemainingMilliseconds $deadlineTicks) -le 0) {
                & $markDeadline
                throw [System.TimeoutException]::new("transport start deadline expired")
            }
            $psi = [System.Diagnostics.ProcessStartInfo]::new()
            $psi.FileName = $SshExecutable
            $psi.UseShellExecute = $false
            $psi.RedirectStandardInput = $true
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.CreateNoWindow = $true
            [void](Set-L04ProcessArguments -ProcessStartInfo $psi -ArgumentList $ArgumentList)
            $process = [System.Diagnostics.Process]::new()
            $process.StartInfo = $psi
            if (-not $process.Start()) {
                throw [System.InvalidOperationException]::new("transport process did not start")
            }
            $state.process_started = $true
            if ((Get-L04RemainingMilliseconds $deadlineTicks) -le 0) {
                & $markDeadline
                throw [System.TimeoutException]::new("transport start deadline expired")
            }
            $stdoutTask = $process.StandardOutput.ReadToEndAsync()
            $stderrTask = $process.StandardError.ReadToEndAsync()
        } catch {
            & $recordException $_.Exception
        }

        if ($state.process_started -and $state.transport_error -eq $null) {
            try {
                if ($BootstrapBytes.Length -gt 0) {
                    $writeTask = $process.StandardInput.BaseStream.WriteAsync($BootstrapBytes, 0, $BootstrapBytes.Length)
                    if (-not (Wait-L04TaskUntilDeadline -Task $writeTask -DeadlineTicks $deadlineTicks)) {
                        & $markDeadline
                        throw [System.TimeoutException]::new("stdin write deadline expired")
                    }
                    $flushTask = $process.StandardInput.BaseStream.FlushAsync()
                    if (-not (Wait-L04TaskUntilDeadline -Task $flushTask -DeadlineTicks $deadlineTicks)) {
                        & $markDeadline
                        throw [System.TimeoutException]::new("stdin flush deadline expired")
                    }
                }
                $process.StandardInput.Close()
                $state.stdin_closed = $true
            } catch {
                Observe-L04Task $writeTask
                Observe-L04Task $flushTask
                try {
                    $process.StandardInput.Close()
                    $state.stdin_closed = $true
                } catch {
                    & $recordException $_.Exception
                }
                & $recordException $_.Exception
            }
        }

        if ($state.process_started) {
            try {
                if (-not $process.HasExited) {
                    $remainingMilliseconds = Get-L04RemainingMilliseconds $deadlineTicks
                    if ($remainingMilliseconds -le 0 -or -not $process.WaitForExit($remainingMilliseconds)) {
                        & $markDeadline
                        throw [System.TimeoutException]::new("transport process deadline expired")
                    }
                }
                if ($process.HasExited) { $state.ssh_exit = $process.ExitCode }
            } catch {
                & $recordException $_.Exception
            }
        }
    } catch {
        & $recordException $_.Exception
    } finally {
        if ($state.deadline_exceeded -or ($state.process_started -and $process -ne $null -and -not $process.HasExited)) {
            $killDeadlineTicks = & $killAndDrainDeadline
            if ($process -ne $null -and $state.process_started) {
                try {
                    if (-not $process.HasExited) {
                        $killTreeMethod = @($process.GetType().GetMethods() | Where-Object {
                            $_.Name -eq "Kill" -and $_.GetParameters().Count -eq 1 -and
                            $_.GetParameters()[0].ParameterType -eq [bool]
                        })
                        if ($killTreeMethod.Count -gt 0) { $process.Kill($true) } else { $process.Kill() }
                    }
                } catch {
                    & $recordException $_.Exception
                }
                try {
                    if (-not $process.HasExited) {
                        $killWaitMilliseconds = Get-L04RemainingMilliseconds $killDeadlineTicks
                        if ($killWaitMilliseconds -gt 0) { [void]$process.WaitForExit($killWaitMilliseconds) }
                    }
                } catch {
                    & $recordException $_.Exception
                }
                try {
                    if ($process.HasExited) {
                        $state.ssh_exit = $process.ExitCode
                    } else {
                        $state.transport_termination_incomplete = $true
                        $state.cleanup_status = "unknown"
                        & $recordError "TerminationIncomplete"
                    }
                } catch {
                    & $recordException $_.Exception
                }
            }
            foreach ($task in @($writeTask, $flushTask)) {
                if ($task -ne $null) {
                    try {
                        if (Wait-L04TaskUntilDeadline -Task $task -DeadlineTicks $killDeadlineTicks) {
                            Observe-L04Task $task
                        } else {
                            & $recordError "StdinTaskDrainTimeout"
                        }
                    } catch {
                        Observe-L04Task $task
                        & $recordException $_.Exception
                    }
                }
            }
            # Publish timeout evidence before potentially blocked inherited stream handles are drained.
            & $publishRawCapture $killDeadlineTicks "timeout"
        }

        if ($process -ne $null -and -not $state.stdin_closed) {
            try {
                $closeDeadline = if ($killDeadlineTicks -ne $null) { $killDeadlineTicks } else { $deadlineTicks }
                if ((Get-L04RemainingMilliseconds $closeDeadline) -le 0) {
                    & $recordError "StdinCloseTimeout"
                } else {
                    $process.StandardInput.Close()
                    $state.stdin_closed = $true
                }
            } catch {
                Observe-L04Task $writeTask
                Observe-L04Task $flushTask
                & $recordException $_.Exception
            }
        }

        $drainDeadline = if ($killDeadlineTicks -ne $null) { $killDeadlineTicks } else { $deadlineTicks }
        foreach ($stream in @(@{ task = $stdoutTask; name = "StdoutDrainTimeout" }, @{ task = $stderrTask; name = "StderrDrainTimeout" })) {
            if ($stream.task -ne $null) {
                try {
                    if (Wait-L04TaskUntilDeadline -Task $stream.task -DeadlineTicks $drainDeadline) {
                        $value = Get-L04CompletedTaskValue -Task $stream.task
                        if ($stream.name -eq "StdoutDrainTimeout") { $state.stdout = [string]$value }
                        else { $state.stderr += [string]$value }
                    } else {
                        & $recordError $stream.name
                    }
                } catch {
                    & $recordException $_.Exception
                }
            }
        }
        $captureDeadline = if ($killDeadlineTicks -ne $null) { $killDeadlineTicks } else { $deadlineTicks }
        & $publishRawCapture $captureDeadline "finalization"
        if ($process -ne $null) {
            try { $process.Dispose() } catch { & $recordException $_.Exception }
        }
        if ($ownedSingleFlightLock) { Exit-L04SingleFlightLock -Lock $SingleFlightLock }
    }

    $rawDigest = $state.raw_capture_sha256
    $rawWritten = $state.raw_capture_write_succeeded -and ($rawDigest -ne $null)
    [pscustomobject]@{
        ssh_exit = $state.ssh_exit
        raw_capture_sha256 = $rawDigest
        raw_capture_written_before_parse = $rawWritten
        raw_capture_write_succeeded = $state.raw_capture_write_succeeded
        transport_error = $state.transport_error
        transport_errors = @($state.transport_errors)
        exception_type = $state.transport_error
        deadline_exceeded = $state.deadline_exceeded
        transport_termination_incomplete = $state.transport_termination_incomplete
        cleanup_status = $state.cleanup_status
        raw_capture_path = if ($state.raw_capture_path -ne $null) { "<raw-capture-path>" } else { $null }
        raw_capture_finalization_error = $state.raw_capture_finalization_error
    }
}

Export-ModuleMember -Function Invoke-L04TransportProcess, Enter-L04SingleFlightLock, Exit-L04SingleFlightLock
