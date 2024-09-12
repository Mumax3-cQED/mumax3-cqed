// Compute new extra term in effective field (see effectivefield.go)
package engine

import (
	"bytes"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	UseCustomKernel = true

	B_rms = NewExcitation("B_rms", "T", "Zero point magnetic field of the cavity")

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

	Wc       float64      = 0 // Resonant frequency of the cavity (rad/s)
	Kappa    float64      = 0 // Cavity dissipation (rad/s)
	X0       float64      = 0 // Initial condition in X-axis
	P0       float64      = 0 // Initial condition in Y-axis
	mem_term *MEMORY_TERM = nil

	HBAR float64 = 1.05457173E-34 // Reduced Planck constant
)

// Equation Memory Term
type MEMORY_TERM struct {
	last_time float64
	csn       [MEMORY_COMPONENTS]float32
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
	mem_term.csn = [MEMORY_COMPONENTS]float32{0, 0}

	// Declaration of new script instructions and functions
	DeclVar("X0", &X0, "Initial condition for the cavity (default=0)")
	DeclVar("P0", &P0, "Initial condition for the cavity (default=0)")
	DeclVar("Kappa", &Kappa, "Cavity dissipation (default=0)")
	DeclVar("Wc", &Wc, "Resonant frequency of the cavity (default=0)")
	DeclVar("HBAR", &HBAR, "Reduced Planck constant")
	DeclVar("UseCustomKernel", &UseCustomKernel, "Use custom CUDA kernel (default=true)")
	DeclFunc("ResetMemoryTerm", ResetMemoryTerm, "Reset memory term for cavity solution")
}

// Adds the current cavity field to dst
func AddCavityField(dst *data.Slice) {

	// start summation from t > 0
	if Time == 0.0 {
		return
	}

	vc2_hbar := (2 * cellVolume()) / HBAR

	brms_slice, rec := B_rms.Slice()
	if rec {
		defer cuda.Recycle(brms_slice)
	}

	full_m := cuda.NewSlice(M.Buffer().NComp(), M.Buffer().Size())
	defer full_m.Free()

	msat_slice, rec := Msat.Slice()
	if rec {
		defer cuda.Recycle(msat_slice)
	}

	mul1N(full_m, msat_slice, M.Buffer())

	dt_time := Time - mem_term.last_time

	cuda.AddCavity(dst, full_m, brms_slice, Wc, Kappa, X0, P0, vc2_hbar, dt_time, Time, &mem_term.csn, Mesh(), UseCustomKernel)

	mem_term.last_time = Time
}

// Reset memory term to start again the cavity calculus
func ResetMemoryTerm() {

	LogIn("")
	LogIn("--------------------------------------------------------------------")
	LogIn("|               Resetting memory... please wait!                   |")
	LogIn("--------------------------------------------------------------------")

	mem_term.Free()

	status := []byte{0, 0}

	if mem_term.last_time == 0.0 {
		LogIn("|           * Init memory component 1... SUCCESS!                  |")
	} else {
		LogIn("|           * Init memory component 1... ERROR!                    |")
		status[0] = 1
	}

	if mem_term.csn[0] == 0 && mem_term.csn[1] == 0 {
		LogIn("|           * Init memory component 2... SUCCESS!                  |")
	} else {
		LogIn("|           * Init memory component 2... ERROR!                    |")
		status[2] = 1
	}

	LogIn("--------------------------------------------------------------------")

	if bytes.ContainsRune(status, 1) {
		LogIn("|            ----> Full memory init... ERROR! <----                |")
	} else {
		LogIn("|            ----> Full memory init... DONE! <----                 |")
	}

	LogIn("--------------------------------------------------------------------")
	LogIn("")
}

// Free memory resources
func (memory *MEMORY_TERM) Free() {
	memory.last_time = 0.0
	memory.csn = [MEMORY_COMPONENTS]float32{0, 0}
}
