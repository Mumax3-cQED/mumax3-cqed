// Compute new extra term in effective field (see effectivefield.go)
package engine

import (
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	DisableBeffContributions = false
	ShowSimulationSummary    = true
	UseCustomKernel          = true

	B_rms = NewExcitation("B_rms", "T", "Zero point magnetic field of the cavity")
	Wc    = NewScalarParam("Wc", "rad/s", "Resonant frequency of the cavity")
	Kappa = NewScalarParam("Kappa", "rad/s", "Cavity dissipation")

	// Read-only variable to check the cavity feature status
	// Calling this variable before setting B_rms in the script will give always 0-status (DISABLED)
	// 1 --> ENABLED (Cavity feature enabled)
	// 0 --> DISABLED (Cavity feature disabled)
	_ = NewScalarValue("CavityFeatureStatus", "", "Check status of cavity feature (1 --> ENABLED, 0 --> DISABLED)", func() float64 {
		status := 0.0
		if IsCavityActive() {
			status = 1.0
		}
		return status
	})

	X0              float64      = 0          // Initial condition in X-axis
	P0              float64      = 0          // Initial condition in Y-axis
	StartCheckpoint time.Time    = time.Now() // Starting date for mumax3 script to measure elapsed execution time, to set starting date anywhere in the  --> StartCheckpoint = now()
	mem_term        *MEMORY_TERM = nil

	HBAR float64 = 1.05457173E-34 // Reduced Planck constant
)

// Equation Memory Term
type MEMORY_TERM struct {
	scn       *data.Slice
	last_time float64
	csn       [MEMORY_COMPONENTS]float64
}

const (
	MEMORY_COMPONENTS = 2
)

// Check whether the cavity feature is active or not by checking the Brms vector,
// If Brms vector is NOT declared in script or Zero-vector the cavity feature is DISABLED otherwise is ENABLED
func IsCavityActive() bool {
	return !B_rms.isZero()
}

// Init memory term
func init() {
	// init new memory term for equation
	mem_term = new(MEMORY_TERM)
	mem_term.last_time = 0.0
	mem_term.csn = [MEMORY_COMPONENTS]float64{0, 0}

	// Declaration of new script instructions and functions
	DeclVar("ShowSimulationSummary", &ShowSimulationSummary, "Show simulation data summary after run() function (default=true)")
	DeclVar("StartCheckpoint", &StartCheckpoint, "Script launch starting date (default now() at the beginning of mumax3 allocation)")
	DeclVar("X0", &X0, "Initial condition for the cavity (default=0)")
	DeclVar("P0", &P0, "Initial condition for the cavity (default=0)")
	DeclVar("HBAR", &HBAR, "Reduced Planck constant")
	DeclVar("UseCustomKernel", &UseCustomKernel, "Use custom CUDA kernel (default=true)")
	DeclVar("DisableBeffContributions", &DisableBeffContributions, "Disables Beff default contributions (default=false)")
	DeclFunc("PrintScriptExecutionTime", PrintScriptExecutionTime, "Print and save to log the script execution time")
	DeclFunc("ResetMemoryTerm", ResetMemoryTerm, "Reset memory term for cavity solution")
}

// Adds the current cavity field to dst
func AddCavityField(dst *data.Slice) {

	// start summation from t > 0
	if Time == 0.0 {
		return
	}

	if UseCustomKernel {
		sizeMesh := Mesh().Size()

		if mem_term.scn != nil && mem_term.scn.Size() != sizeMesh {
			mem_term.Free()
		}

		if mem_term.scn == nil {
			mem_term.scn = cuda.NewSlice(MEMORY_COMPONENTS, sizeMesh)
			mem_term.last_time = 0.0
		}
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

	dt_time := Time - mem_term.last_time

	cuda.AddCavity(dst, M.Buffer(), brms_slice, mem_term.scn, wc_slice, kappa, X0, P0, msatCell, dt_time, Time, &mem_term.csn, Mesh(), UseCustomKernel)

	mem_term.last_time = Time
}
