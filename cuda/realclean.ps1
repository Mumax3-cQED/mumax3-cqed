Get-Childitem . -recurse -force -include *_wrapper.go,*.ptx | remove-item -force
remove-item -force cuda2go.exe
