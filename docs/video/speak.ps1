param([string]$NarrationJson, [string]$OutDir, [string]$VoiceLike = "Mark")
# One WAV per sentence with a OneCore voice through WinRT (falls back to System.Speech).
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$items = Get-Content -Raw -Encoding UTF8 $NarrationJson | ConvertFrom-Json
$useWinRT = $true
try {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
    $null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
    $asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
    function Await($op, $type) { $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); $null = $t.Wait(-1); $t.Result }
    $synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
    $voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.DisplayName -like "*$VoiceLike*" } | Select-Object -First 1
    if ($voice) { $synth.Voice = $voice }
    $synth.Options.SpeakingRate = 1.03
    Write-Output ("voice: " + $synth.Voice.DisplayName)
} catch { $useWinRT = $false; Write-Output ("WinRT unavailable: " + $_.Exception.Message) }

$i = 0
foreach ($it in $items) {
    $path = Join-Path $OutDir ("s{0:D2}.wav" -f $i)
    if ($useWinRT) {
        $stream = Await ($synth.SynthesizeTextToStreamAsync($it.say)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
        $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
        $size = [uint32]$stream.Size
        $null = Await ($reader.LoadAsync($size)) ([uint32])
        $bytes = New-Object byte[] $size
        $reader.ReadBytes($bytes)
        [IO.File]::WriteAllBytes($path, $bytes)
    } else {
        Add-Type -AssemblyName System.Speech
        $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
        $s.SelectVoice("Microsoft David Desktop"); $s.SetOutputToWaveFile($path); $s.Speak($it.say); $s.Dispose()
    }
    $i++
}
Write-Output ("wrote $i sentences to $OutDir")
