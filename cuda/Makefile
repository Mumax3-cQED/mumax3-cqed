.PHONY: all clean realclean

all:
	./make.bash
	go install -v "github.com/mumax/3/..."

cuda2go: cuda2go.go
	go build $<

clean:
	rm -vf *.ptx

realclean:
	rm -vf *_wrapper.go *.ptx cuda2go
