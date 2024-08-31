// Compute new extra term in effective field (see effectivefield.go)
package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Adds the current cavity field to dst
func AddCavityField(dst *data.Slice) {

	// start summation from t > 0
	if Time == 0.0 {
		return
	}

	sizeMesh := Mesh().Size()

	if UseCustomKernel && mem_term.scn != nil && mem_term.scn.Size() != sizeMesh {
		mem_term.Free()
	}

	if UseCustomKernel && mem_term.scn == nil {
		mem_term.scn = cuda.NewSlice(MEMORY_COMPONENTS, sizeMesh)
		mem_term.last_time = 0.0
		mem_term.dt_time = 0.0
	}

	msatCell := calcMsatCellVol()

	wc_slice := Wc.MSlice()
	defer wc_slice.Recycle()

	brms_slice, rec := B_rms.Slice()
	if rec {
		defer cuda.Recycle(brms_slice)
	}

	kappa := Kappa.MSlice()
	defer kappa.Recycle()

	mem_term.dt_time = Time - mem_term.last_time

	if UseCustomKernel {
		cuda.AddCavity(dst, M.Buffer(), mem_term.scn, brms_slice, wc_slice, kappa, X0, P0, msatCell, mem_term.dt_time, Time, Mesh())
	} else {
		cuda.AddCavity2(dst, M.Buffer(), brms_slice, wc_slice, kappa, X0, P0, msatCell, mem_term.dt_time, Time, &mem_term.csn, Mesh())
	}

	mem_term.last_time = Time
}
