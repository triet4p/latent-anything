[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [Alias("SshPath")] [string]$SshExecutable,
    [Parameter(Mandatory = $true)] [string]$RemoteTarget,
    [Parameter(Mandatory = $true)] [string]$PayloadPath,
    [Parameter(Mandatory = $true)]
    [ValidateSet("IntegratedGradients", "TCAV", "DirectLogitLens", "TunedLogitLens", "Disentanglement", "TrueActivationPatching", "AdditiveSteering")]
    [string]$UseCase,
    [Parameter(Mandatory = $true)] [string]$CodeSha,
    [Parameter(Mandatory = $true)] [string]$RepoUrl,
    [Parameter(Mandatory = $true)] [string]$RawCapturePath,
    [ValidateRange(2400, 7200)] [int]$TransportTimeoutSeconds = 3600,
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
    return ([Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($Bytes))).ToLowerInvariant()
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
}

function New-Bootstrap {
    param([string]$PayloadBase64, [string]$PayloadSha256)
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
bash "$DecodedPayload" "$@"
semantic_status=$?
exit "$semantic_status"
'@
    $cr = [string][char]13; $lf = [string][char]10
    return $bootstrap.Replace("__PAYLOAD_SHA256__", $PayloadSha256).Replace("__PAYLOAD_BASE64__", $PayloadBase64).Replace($cr + $lf, $lf).Replace($cr, $lf)
}

Assert-TransportParameters
$normalizedCodeSha = $CodeSha.ToLowerInvariant()
$payloadBytes = Get-Utf8LfBytes ([System.IO.File]::ReadAllBytes($PayloadPath))
$payloadSha256 = Get-Sha256Hex $payloadBytes
$payloadBase64 = [Convert]::ToBase64String($payloadBytes)
$bootstrapBytes = [System.Text.UTF8Encoding]::new($false).GetBytes((New-Bootstrap -PayloadBase64 $payloadBase64 -PayloadSha256 $payloadSha256))
$bootstrapSha256 = Get-Sha256Hex $bootstrapBytes
$manifest = [ordered]@{
    schema_version = "m14-l04-remote-transport-build-v1"
    mode = if ($BuildOnly -or $DryRunMode) { "build-only" } else { "execute" }
    use_case = $UseCase; code_sha = $normalizedCodeSha
    payload = [ordered]@{ sha256 = $payloadSha256; bytes = $payloadBytes.Length }
    bootstrap = [ordered]@{ sha256 = $bootstrapSha256; bytes = $bootstrapBytes.Length }
    transport_timeout_seconds = $TransportTimeoutSeconds
    kill_grace_seconds = 30
    expected_markers = @("L04_TRANSPORT_PAYLOAD_SHA256", "L04_TRANSPORT_DECODE_STATUS", "L04_TRANSPORT_DECODE_SHA256", "L04_TRANSPORT_DECODE_MATCH", "L04_WORKDIR", "L04_USE_CASE", "L04_CODE_SHA", "L04_CLI_STATUS", "L04_BUNDLE_STATUS", "L04_STATUS", "L04_BUNDLE_BYTES", "L04_BUNDLE_SHA256", "L04_BUNDLE_MEMBER", "L04_BUNDLE_B64_BEGIN", "L04_BUNDLE_B64_END", "L04_CLEANUP", "L04_TRANSPORT_CLEANUP")
    command_args_redacted = @("<ssh.exe>", "<remote-target>", "bash", "-s", "--", "<use-case>", "<code-sha>", "<repo-url>")
    secrets_redacted = $true; raw_capture_path_redacted = "<raw-capture-path>"
}
if ($BuildOnly -or $DryRunMode) { $manifest | ConvertTo-Json -Depth 8 -Compress; exit 0 }

$seamPath = Join-Path $PSScriptRoot "_m14_l04_transport_seam.psm1"
Import-Module $seamPath -Force
$capture = Invoke-L04TransportProcess -SshExecutable $SshExecutable -ArgumentList @($RemoteTarget, "bash", "-s", "--", $UseCase, $normalizedCodeSha, $RepoUrl) -BootstrapBytes $bootstrapBytes -RawCapturePath $RawCapturePath -TimeoutSeconds $TransportTimeoutSeconds
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
if ($capture.transport_error -ne $null) { exit 70 }
if ($Postprocess -and $capture.raw_capture_write_succeeded) {
    if ([string]::IsNullOrWhiteSpace($AuditOutputPath)) {
        $AuditOutputPath = Join-Path $ArtifactOutputDir ("l04-explanations.ssh.$UseCase.$normalizedCodeSha.audit.json")
    }
    & uv run python -m scripts.m14_l04_remote_postprocess `
        --retain `
        --raw-capture $RawCapturePath `
        --source-sha $normalizedCodeSha `
        --use-case $UseCase `
        --artifact-dir $ArtifactOutputDir `
        --audit $AuditOutputPath
    $postprocessExit = $LASTEXITCODE
    if ($postprocessExit -ne 0) { exit $postprocessExit }
}
exit $capture.ssh_exit
