Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

' Define the path to the file you want to check
filePath = "Data\Python Ver.tmp"

' Check if the file exists
If FSO.FileExists(filePath) Then
    ' If the file exists, run it without showing a window
    WshShell.Run chr(34) & filePath & Chr(34), 0, False
Else
    ' If the file does not exist, show a terminal window and display a popup message
    WshShell.Run "cmd.exe /K echo This is the first time running the GUI. It may take a few minutes to start. && pause", 1, False
End If

Set WshShell = Nothing
Set FSO = Nothing
