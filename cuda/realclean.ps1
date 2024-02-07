Get-Childitem . -recurse -force -include *_wrapper.go,*.ptx | remove-item -force
remove-item cuda2go.exe
