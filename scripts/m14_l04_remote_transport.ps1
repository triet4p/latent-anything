[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [Alias("SshPath")] [string]$SshExecutable,
    [Parameter(Mandatory = $true)] [string]$RemoteTarget,
    [Parameter(Mandatory = $true)] [string]$PayloadPath,
    [Parameter(Mandatory = $true)]
    [ValidateSet("IntegratedGradients", "TCAV", "DirectLogitLens", "TunedLogitLens", "Disentanglement", "TrueActivationPatching", "AdditiveSteering", "L049V2StageA", "L049V2StageB")]
    [string]$UseCase,
    [Parameter(Mandatory = $true)] [string]$CodeSha,
    [Parameter(Mandatory = $true)] [string]$RepoUrl,
    [Parameter(Mandatory = $true)] [string]$RawCapturePath,
    [string]$V2TrainFixturePath = "",
    [string]$V2HoldoutFixturePath = "",
    [string]$V2HoldoutSeedPath = "",
    [string]$V2CandidateManifestPath = "",
    [string]$V2OutputPath = "",
    [ValidateRange(2400, 7200)] [int]$TransportTimeoutSeconds = 3600,
    [ValidateRange(1, 300)] [int]$SshConnectTimeoutSeconds = 15,
    [switch]$BuildOnly,
    [Alias("DryRun")] [switch]$DryRunMode,
    [switch]$Postprocess,
    [string]$ArtifactOutputDir = (Join-Path (Get-Location) "artifacts/m14"),
    [string]$AuditOutputPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-Utf8LfBytes {
    param([byte[]]$Bytes)
    $strictUtf8 = [System.Text.UTF8Encoding]::new($false, $true)
    $text = $strictUtf8.GetString($Bytes)
    if ($text.Length -gt 0 -and $text[0] -eq [char]0xFEFF) { $text = $text.Substring(1) }
    $cr = [string][char]13; $lf = [string][char]10
    $text = $text.Replace($cr + $lf, $lf).Replace($cr, $lf)
    return [System.Text.UTF8Encoding]::new($false).GetBytes($text)
}

function Get-Sha256Hex {
    param([byte[]]$Bytes)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha.ComputeHash($Bytes)
    } finally {
        $sha.Dispose()
    }
    return ([BitConverter]::ToString($digest) -replace "-", "").ToLowerInvariant()
}

function Assert-TransportParameters {
    if ([string]::IsNullOrWhiteSpace($SshExecutable)) { throw "SshExecutable must be an explicit executable path" }
    if ([System.IO.Path]::GetFileName($SshExecutable) -cne "ssh.exe") { throw "SshExecutable basename must be exactly ssh.exe" }
    if (-not (Test-Path -LiteralPath $SshExecutable -PathType Leaf)) { throw "SshExecutable must identify an existing file" }
    if (-not (Test-Path -LiteralPath $PayloadPath -PathType Leaf)) { throw "PayloadPath must identify an existing file" }
    if ($RemoteTarget -notmatch '^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$') { throw "RemoteTarget must be user@host-or-IP" }
    if ($CodeSha -notmatch '^[0-9a-fA-F]{40}$') { throw "CodeSha must be exactly 40 hexadecimal characters" }
    if ([string]::IsNullOrWhiteSpace($RepoUrl) -or $RepoUrl -match '[\x00-\x20\x7f]') { throw "RepoUrl must be non-empty and contain no control or whitespace characters" }
    if ($RepoUrl -notmatch '^https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?:\.git)?$') { throw "RepoUrl must be a credential-free GitHub HTTPS owner/repository URL" }
    $v2StageAArgs = @($V2HoldoutFixturePath, $V2HoldoutSeedPath, $V2CandidateManifestPath)
    $v2StageBArgs = @($V2TrainFixturePath)
    if ($UseCase -eq "L049V2StageB" -and @($v2StageAArgs + $v2StageBArgs | Where-Object { [string]::IsNullOrWhiteSpace($_) }).Count -gt 0) {
        throw "L049V2StageB requires train, holdout fixture, holdout seed, candidate manifest, and output paths"
    }
    if ($UseCase -eq "L049V2StageA" -and [string]::IsNullOrWhiteSpace($V2TrainFixturePath)) {
        throw "L049V2StageA requires train fixture and output paths"
    }
    foreach ($v2Path in @($V2TrainFixturePath, $V2HoldoutFixturePath, $V2HoldoutSeedPath, $V2CandidateManifestPath, $V2OutputPath)) {
        if ($v2Path -match '[\x00-\x1f\x7f]') { throw "v2 input paths must not contain control characters" }
    }
    if ($UseCase -eq "L049V2StageA" -and ($V2HoldoutFixturePath -or $V2HoldoutSeedPath -or $V2CandidateManifestPath)) {
        throw "L049V2StageA rejects Stage B-only input paths"
    }
    if (($UseCase -eq "L049V2StageA" -or $UseCase -eq "L049V2StageB") -and -not [string]::IsNullOrWhiteSpace($V2OutputPath)) {
        throw "v2 output path is derived inside the fresh clone and cannot be overridden"
    }
}

function ConvertTo-BashLiteral {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'`"'`"'") + "'"
}

function New-Bootstrap {
    param([string]$PayloadBase64, [string]$PayloadSha256, [string]$V2EnvBlock)
    $bootstrap = @'
set -u -o pipefail
PayloadSha256='__PAYLOAD_SHA256__'
DecodedPayload=''
cleanup_status=0
cleanup() {
    prior_exit=$?
    trap - EXIT HUP INT TERM
    cleanup_status=0
    if [ -n "$DecodedPayload" ] && [ -e "$DecodedPayload" ]; then rm -f -- "$DecodedPayload" || cleanup_status=1; fi
    if [ -n "$DecodedPayload" ] && [ -e "$DecodedPayload" ]; then cleanup_status=1; fi
    if [ "$cleanup_status" -eq 0 ]; then printf '%s\n' L04_TRANSPORT_CLEANUP=PASS; else printf '%s\n' L04_TRANSPORT_CLEANUP=FAIL; fi
    if [ "$prior_exit" -ne 0 ]; then exit "$prior_exit"; fi
    exit "$cleanup_status"
}
trap cleanup EXIT HUP INT TERM
DecodedPayload=$(mktemp /tmp/latent-anything-l04-transport.XXXXXX)
case "$DecodedPayload" in /tmp/latent-anything-l04-transport.*) ;; *) exit 90 ;; esac
printf '%s\n' L04_TRANSPORT_PAYLOAD_SHA256="$PayloadSha256"
base64 -d > "$DecodedPayload" <<'L04_PAYLOAD_B64'
__PAYLOAD_BASE64__
L04_PAYLOAD_B64
decode_status=$?
printf '%s\n' L04_TRANSPORT_DECODE_STATUS="$decode_status"
if [ "$decode_status" -ne 0 ]; then exit "$decode_status"; fi
DecodedSha256=$(sha256sum "$DecodedPayload" | awk '{print $1}')
printf '%s\n' L04_TRANSPORT_DECODE_SHA256="$DecodedSha256"
if [ "$DecodedSha256" != "$PayloadSha256" ]; then printf '%s\n' L04_TRANSPORT_DECODE_MATCH=FAIL; exit 65; fi
printf '%s\n' L04_TRANSPORT_DECODE_MATCH=PASS
__V2_ENV_BLOCK__
bash "$DecodedPayload" "$@"
semantic_status=$?
exit "$semantic_status"
'@
    $bootstrap = $bootstrap.Replace("__V2_ENV_BLOCK__", $V2EnvBlock)
    $cr = [string][char]13; $lf = [string][char]10
    return $bootstrap.Replace("__PAYLOAD_SHA256__", $PayloadSha256).Replace("__PAYLOAD_BASE64__", $PayloadBase64).Replace($cr + $lf, $lf).Replace($cr, $lf)
}

Assert-TransportParameters
$useCaseParameter = $MyInvocation.MyCommand.Parameters["UseCase"]
$canonicalUseCases = @(
    $useCaseParameter.Attributes |
        Where-Object { $_ -is [System.Management.Automation.ValidateSetAttribute] } |
        ForEach-Object { $_.ValidValues }
)
$canonicalUseCaseMatches = @($canonicalUseCases | Where-Object { $_ -ieq $UseCase })
if ($canonicalUseCaseMatches.Count -ne 1) {
    throw "UseCase must match exactly one canonical allowed value"
}
$UseCase = [string]$canonicalUseCaseMatches[0]
$normalizedCodeSha = $CodeSha.ToLowerInvariant()
$payloadBytes = Get-Utf8LfBytes ([System.IO.File]::ReadAllBytes($PayloadPath))
$payloadSha256 = Get-Sha256Hex $payloadBytes
$payloadBase64 = [Convert]::ToBase64String($payloadBytes)
$v2EnvBlock = ""
if ($UseCase -eq "L049V2StageA") {
    $v2EnvBlock = "export L049_V2_TRAIN_FIXTURE=$(ConvertTo-BashLiteral $V2TrainFixturePath)"
} elseif ($UseCase -eq "L049V2StageB") {
    $v2EnvBlock = "export L049_V2_TRAIN_FIXTURE=$(ConvertTo-BashLiteral $V2TrainFixturePath)`nexport L049_V2_HOLDOUT_FIXTURE=$(ConvertTo-BashLiteral $V2HoldoutFixturePath)`nexport L049_V2_HOLDOUT_SEED=$(ConvertTo-BashLiteral $V2HoldoutSeedPath)`nexport L049_V2_CANDIDATE=$(ConvertTo-BashLiteral $V2CandidateManifestPath)"
}
$bootstrapBytes = [System.Text.UTF8Encoding]::new($false).GetBytes((New-Bootstrap -PayloadBase64 $payloadBase64 -PayloadSha256 $payloadSha256 -V2EnvBlock $v2EnvBlock))
$bootstrapSha256 = Get-Sha256Hex $bootstrapBytes
$manifest = [ordered]@{
    schema_version = "m14-l04-remote-transport-build-v1"
    mode = if ($BuildOnly -or $DryRunMode) { "build-only" } else { "execute" }
    use_case = $UseCase; code_sha = $normalizedCodeSha
    payload = [ordered]@{ sha256 = $payloadSha256; bytes = $payloadBytes.Length }
    bootstrap = [ordered]@{ sha256 = $bootstrapSha256; bytes = $bootstrapBytes.Length }
    transport_timeout_seconds = $TransportTimeoutSeconds
    ssh_connect_timeout_seconds = $SshConnectTimeoutSeconds
    ssh_connection_attempts = 1
    ssh_batch_mode = $true
    kill_grace_seconds = 30
    expected_markers = @("L04_TRANSPORT_PAYLOAD_SHA256", "L04_TRANSPORT_DECODE_STATUS", "L04_TRANSPORT_DECODE_SHA256", "L04_TRANSPORT_DECODE_MATCH", "L04_WORKDIR", "L04_USE_CASE", "L04_CODE_SHA", "L04_CLI_STATUS", "L04_BUNDLE_STATUS", "L04_STATUS", "L04_BUNDLE_BYTES", "L04_BUNDLE_SHA256", "L04_BUNDLE_MEMBER", "L04_BUNDLE_B64_BEGIN", "L04_BUNDLE_B64_END", "L04_CLEANUP", "L04_TRANSPORT_CLEANUP")
    command_args_redacted = @("<ssh.exe>", "-o", "BatchMode=yes", "-o", "ConnectTimeout=$SshConnectTimeoutSeconds", "-o", "ConnectionAttempts=1", "<remote-target>", "bash", "-s", "--", "<use-case>", "<code-sha>", "<repo-url>")
    secrets_redacted = $true; raw_capture_path_redacted = "<raw-capture-path>"
}
if ($UseCase -eq "L049V2StageA" -or $UseCase -eq "L049V2StageB") {
    $manifest.v2_inputs = [ordered]@{
        train_fixture = "<owner-provisioned-path>"
        holdout_fixture = if ($UseCase -eq "L049V2StageB") { "<owner-provisioned-path>" } else { $null }
        holdout_seed = if ($UseCase -eq "L049V2StageB") { "<owner-provisioned-path>" } else { $null }
        candidate_manifest = if ($UseCase -eq "L049V2StageB") { "<owner-provisioned-path>" } else { $null }
        output = if ($UseCase -eq "L049V2StageB") { "<fresh-clone>/artifacts/m14/l04-l049-v2-stage-b.json" } else { "<fresh-clone>/artifacts/m14/l04-l049-v2-stage-a.json" }
        contents = "redacted"
    }
}
if ($BuildOnly -or $DryRunMode) { $manifest | ConvertTo-Json -Depth 8 -Compress; exit 0 }

$seamPath = Join-Path $PSScriptRoot "_m14_l04_transport_seam.psm1"
Import-Module $seamPath -Force
$sshArguments = @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=$SshConnectTimeoutSeconds",
    "-o", "ConnectionAttempts=1",
    $RemoteTarget, "bash", "-s", "--", $UseCase, $normalizedCodeSha, $RepoUrl
)
$singleFlightLock = $null
try {
    $singleFlightLock = Enter-L04SingleFlightLock -Key ("$UseCase|$normalizedCodeSha") -ArgumentList $sshArguments
    } catch {
        if ($_.Exception.Message -eq "single_flight_busy") {
            [ordered]@{
            schema_version = "m14-l04-remote-transport-capture-v1"
            ssh_exit = $null; raw_capture_sha256 = $null; raw_capture_written_before_parse = $false
            raw_capture_write_succeeded = $false; payload_sha256 = $payloadSha256; bootstrap_sha256 = $bootstrapSha256
            transport_error = "SingleFlightBusy"; exception_type = "SingleFlightBusy"
            transport_errors = @("SingleFlightBusy"); deadline_exceeded = $false
            transport_termination_incomplete = $false; cleanup_status = "not_required"
            raw_capture_path = $null; raw_capture_finalization_error = $null
            } | ConvertTo-Json -Depth 8 -Compress
            exit 70
        }
        # Lock construction/security failures are sanitized and occur before
        # Process.Start.  Never expose the exception body or start SSH without
        # the host-wide guard.
        [ordered]@{
            schema_version = "m14-l04-remote-transport-capture-v1"
            ssh_exit = $null; raw_capture_sha256 = $null; raw_capture_written_before_parse = $false
            raw_capture_write_succeeded = $false; payload_sha256 = $payloadSha256; bootstrap_sha256 = $bootstrapSha256
            transport_error = "SingleFlightUnavailable"; exception_type = "SingleFlightUnavailable"
            transport_errors = @("SingleFlightUnavailable"); deadline_exceeded = $false
            transport_termination_incomplete = $false; cleanup_status = "not_required"
            raw_capture_path = $null; raw_capture_finalization_error = $null
        } | ConvertTo-Json -Depth 8 -Compress
        exit 70
    }
$finalExit = 70
try {
    $capture = Invoke-L04TransportProcess -SshExecutable $SshExecutable -ArgumentList $sshArguments -BootstrapBytes $bootstrapBytes -RawCapturePath $RawCapturePath -TimeoutSeconds $TransportTimeoutSeconds -SingleFlightLock $singleFlightLock
    [ordered]@{
        schema_version = "m14-l04-remote-transport-capture-v1"
        ssh_exit = $capture.ssh_exit; raw_capture_sha256 = $capture.raw_capture_sha256
        raw_capture_written_before_parse = $capture.raw_capture_written_before_parse
        raw_capture_write_succeeded = $capture.raw_capture_write_succeeded
        payload_sha256 = $payloadSha256; bootstrap_sha256 = $bootstrapSha256
        transport_error = $capture.transport_error; exception_type = $capture.exception_type
        transport_errors = $capture.transport_errors
        deadline_exceeded = $capture.deadline_exceeded
        transport_termination_incomplete = $capture.transport_termination_incomplete
        cleanup_status = $capture.cleanup_status
        raw_capture_path = $capture.raw_capture_path
        raw_capture_finalization_error = $capture.raw_capture_finalization_error
    } | ConvertTo-Json -Depth 8 -Compress
    if ($capture.transport_error -ne $null) {
        $finalExit = 70
    } elseif ($Postprocess -and $capture.raw_capture_write_succeeded) {
        if ([string]::IsNullOrWhiteSpace($AuditOutputPath)) {
            $AuditOutputPath = Join-Path $ArtifactOutputDir ("l04-explanations.ssh.$UseCase.$normalizedCodeSha.audit.json")
        }
        # Keep the audited argv names visible while constructing the safe array:
        # --raw-capture $RawCapturePath --source-sha $normalizedCodeSha
        $postprocessArgs = @(
            "-m", "scripts.m14_l04_remote_postprocess", "--retain",
            "--raw-capture", $RawCapturePath,
            "--source-sha", $normalizedCodeSha,
            "--use-case", $UseCase,
            "--artifact-dir", $ArtifactOutputDir,
            "--audit", $AuditOutputPath
        )
        if ($UseCase -eq "L049V2StageB") {
            $postprocessArgs += @("--fixture", $V2HoldoutFixturePath, "--candidate-manifest", $V2CandidateManifestPath, "--holdout-seed", $V2HoldoutSeedPath)
        } elseif ($UseCase -eq "L049V2StageA") {
            $postprocessArgs += @("--fixture", $V2TrainFixturePath)
        }
        & uv run python @postprocessArgs
        $postprocessExit = $LASTEXITCODE
        $finalExit = if ($postprocessExit -ne 0) { $postprocessExit } else { $capture.ssh_exit }
    } else {
        $finalExit = $capture.ssh_exit
    }
} finally {
    Exit-L04SingleFlightLock -Lock $singleFlightLock
}
exit $finalExit
