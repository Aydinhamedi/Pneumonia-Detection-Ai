Set WshShell = CreateObject("WScript.Shell")
WshShell.Run chr(34) & "GUI.cmd" & Chr(34), 0
Set WshShell = Nothing