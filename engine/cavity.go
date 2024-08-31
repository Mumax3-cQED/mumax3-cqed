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

	cuda.AddCavity(dst, M.Buffer(), brms_slice, wc_slice, kappa, X0, P0, msatCell, mem_term.dt_time, Time, &mem_term.csn, Mesh(), UseCustomKernel)

	mem_term.last_time = Time
}
