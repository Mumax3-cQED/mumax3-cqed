Get-Childitem . -recurse -force -include *_wrapper.go,*.ptx | Remove-Item -force
Remove-Item -force cuda2go.exe
