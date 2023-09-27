.PHONY: all clean realclean

all:
	./make.bash
	go install -v

cuda2go: cuda2go.go
	go build $<

clean:
	rm -vf *.ptx

realclean:
	rm -vf *_wrapper.go *.ptx cuda2go
