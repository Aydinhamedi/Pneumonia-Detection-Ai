Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

' Define the path to the file you want to check
filePath = "Data\Python Ver.tmp"

' Check if the file exists
If FSO.FileExists(filePath) Then
    ' If the file exists, run it without showing a window
    WshShell.Run chr(34) & "GUI.cmd" & Chr(34), 0, False
Else
    ' If the file does not exist, run GUI.cmd with a terminal window and show a popup message
    WshShell.Run chr(34) & "GUI.cmd" & Chr(34), 1, False
    WshShell.Popup "This is the first time running the GUI. It may take a few minutes to start.", 30, "First Time Running", 64
End If

Set WshShell = Nothing
Set FSO = Nothing
