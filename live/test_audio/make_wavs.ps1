Add-Type -AssemblyName System.Speech
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$items = @{
  "live_part.wav"     = "What is this part?";
  "live_tank.wav"     = "What is the capacity of the lubrication tank?";
  "live_weather.wav"  = "What's the weather today?";
  "esp_nova_tank.wav" = "Nova, what is the capacity of the lubrication tank?";
  "esp_nova_look.wav" = "Nova, what is this part?"
}
foreach ($k in $items.Keys) {
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  $s.SetOutputToWaveFile((Join-Path $PSScriptRoot $k), $fmt)
  $s.Speak($items[$k]); $s.Dispose()
}
Get-ChildItem $PSScriptRoot -Filter *.wav | Select Name, Length
