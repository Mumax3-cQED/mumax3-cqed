package engine

// MODIFIED INMA
import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Alpha                              = NewScalarParam("alpha", "", "Landau-Lifshitz damping constant")
	Xi                                 = NewScalarParam("xi", "", "Non-adiabaticity of spin-transfer-torque")
	Pol                                = NewScalarParam("Pol", "", "Electrical current polarization")
	Lambda                             = NewScalarParam("Lambda", "", "Slonczewski Λ parameter")
	EpsilonPrime                       = NewScalarParam("EpsilonPrime", "", "Slonczewski secondairy STT term ε'")
	FrozenSpins                        = NewScalarParam("frozenspins", "", "Defines spins that should be fixed") // 1 - frozen, 0 - free. TODO: check if it only contains 0/1 values
	FixedLayer                         = NewExcitation("FixedLayer", "", "Slonczewski fixed layer polarization")
	Torque                             = NewVectorField("torque", "T", "Total torque/γ0", SetTorque)
	LLTorque                           = NewVectorField("LLtorque", "T", "Landau-Lifshitz torque/γ0", SetLLTorque)
	STTorque                           = NewVectorField("STTorque", "T", "Spin-transfer torque/γ0", AddSTTorque)
	FreeLayerThickness                 = NewScalarParam("FreeLayerThickness", "m", "Slonczewski free layer thickness (if set to zero (default), then the thickness will be deduced from the mesh size)")
	J                                  = NewExcitation("J", "A/m2", "Electrical current density")
	MaxTorque                          = NewScalarValue("maxTorque", "T", "Motion term for LLG equation", GetMaxTorque)
	GammaLL                    float64 = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	Precess                            = true
	DisableZhangLiTorque               = false
	DisableSlonczewskiTorque           = false
	DisableTimeEvolutionTorque         = true
	DisableBeffContributions           = false
	EnableCavityDissipation            = false
	fixedLayerPosition                 = FIXEDLAYER_TOP // instructs mumax3 how free and fixed layers are stacked along +z direction

	start  time.Time
	B_rms          = NewExcitation("B_rms", "T", "Zero point magnetic field of the cavity")
	Wc             = NewScalarParam("Wc", "rad/s", "Resonant frequency of the cavity")
	Kappa          = NewScalarParam("Kappa", "rad/s", "Cavity dissipation")
	NSpins float64 = 0 // Number of spins

	mem_term *MEMORY_TERM = nil
)

const (
	MuB               = 9.2740091523E-24
	MEMORY_COMPONENTS = 2
)

// Memory term computation
type MEMORY_TERM struct {
	scn       *data.Slice
	last_time float64
	dt_time   float64
}

func init() {

	start = time.Now()
	mem_term = new(MEMORY_TERM)  // init new memory term for equation
	Pol.setUniform([]float64{1}) // default spin polarization
	Lambda.Set(1)                // sensible default value (?).

	DeclVar("NSpins", &NSpins, "Number of spins")
	DeclVar("GammaLL", &GammaLL, "Gyromagnetic ratio in rad/Ts")
	DeclVar("DisableZhangLiTorque", &DisableZhangLiTorque, "Disables Zhang-Li torque (default=false)")
	DeclVar("DisableSlonczewskiTorque", &DisableSlonczewskiTorque, "Disables Slonczewski torque (default=false)")
	DeclVar("DisableTimeEvolutionTorque", &DisableTimeEvolutionTorque, "Disables Cavity Time evolution torque (default=true)")
	DeclVar("DisableBeffContributions", &DisableBeffContributions, "Disables Beff default contributions (default=false)")
	DeclVar("EnableCavityDissipation", &EnableCavityDissipation, "Enable/Disable Cavity Dissipation (default=false)")
	DeclVar("DoPrecess", &Precess, "Enables LL precession (default=true)")
	DeclLValue("FixedLayerPosition", &flposition{}, "Position of the fixed layer: FIXEDLAYER_TOP, FIXEDLAYER_BOTTOM (default=FIXEDLAYER_TOP)")
	DeclROnly("FIXEDLAYER_TOP", FIXEDLAYER_TOP, "FixedLayerPosition = FIXEDLAYER_TOP instructs mumax3 that fixed layer is on top of the free layer")
	DeclROnly("FIXEDLAYER_BOTTOM", FIXEDLAYER_BOTTOM, "FixedLayerPosition = FIXEDLAYER_BOTTOM instructs mumax3 that fixed layer is underneath of the free layer")
	DeclFunc("PrintScriptExecutionTime", PrintScriptExecutionTime, "Print and save to log the script execution time")
}

func PrintScriptExecutionTime() {

	_, months, days, hours, mins, secs := getTimeDifference(start)
	full_diff := ""

	if months > 0 {
		full_diff += fmt.Sprintf("%dM;", months)
	}

	if days > 0 {
		full_diff += fmt.Sprintf("%dd;", days)
	}

	if hours > 0 {
		full_diff += fmt.Sprintf("%dh;", hours)
	}

	if mins > 0 {
		full_diff += fmt.Sprintf("%dm;", mins)
	}

	if secs > 0 {
		full_diff += fmt.Sprintf("%ds", secs)
	}

	full_diff = strings.Replace(full_diff, ";", " ", -1)

	LogIn("\n ---> Full mumax3 script running time:", full_diff, "\n")
}

// Display a script configuration summary and log the information into the log.txt file
func PrintParametersTimeEvolution(simulationTime *float64) {

	if !DisableTimeEvolutionTorque {

		// check if not empty
		if mem_term.scn != nil {
			mem_term.Free()
		}

		c, rec := B_rms.Slice()
		if rec {
			defer cuda.Recycle(c)
		}

		be, rec := B_ext.Slice()
		if rec {
			defer cuda.Recycle(be)
		}

		v := Wc.MSlice()
		defer v.Recycle()

		m_sat := Msat.MSlice()
		defer m_sat.Recycle()

		alpha := Alpha.MSlice()
		defer alpha.Recycle()

		LogIn("")
		LogIn("------------------------------------------------")

		year, month, day, hour, minute, seconds := getCurrentDate()
		full_date := fmt.Sprintf("%d-%02d-%02d %02d:%02d:%02d", year, month, day, hour, minute, seconds)

		LogIn(" Simulation date (yyyy-MM-dd HH:mm:ss):", full_date)
		LogIn(" Time evolution factor in LLG equation: Enabled")

		if DisableBeffContributions {
			LogIn(" Beff default contributions: Disabled")
		} else {
			LogIn(" Beff default contributions: Enabled")
		}

		if EnableDemag {
			LogIn(" B_demag: Enabled")
		} else {
			LogIn(" B_demag: Disabled")
		}

		if DisableZhangLiTorque {
			LogIn(" Zhang-Li Spin-Transfer Torque: Disabled")
		} else {
			LogIn(" Zhang-Li Spin-Transfer Torque: Enabled")
		}

		if DisableSlonczewskiTorque {
			LogIn(" Slonczewski Spin-Transfer Torque: Disabled")
		} else {
			LogIn(" Slonczewski Spin-Transfer Torque: Enabled")
		}

		if EnableCavityDissipation {
			LogIn(" Cavity Dissipation: Enabled")

			kappa := Kappa.MSlice()
			defer kappa.Recycle()

			LogIn(" Kappa (rad/s):", kappa.Mul(0))
		} else {
			LogIn(" Cavity Dissipation: Disabled")
		}

		full_sizex, full_sizey, full_sizez, cell_size, num_cells := calcFullSize()

		LogIn(" Shape size (m):", full_sizex, "x", full_sizey, "x", full_sizez)
		LogIn(" Num. cells:", num_cells[X], "x", num_cells[Y], "x", num_cells[Z])
		LogIn(" Cell size (m):", cell_size[X], "x", cell_size[Y], "x", cell_size[Z])

		if alpha.Mul(0) != 0 {
			LogIn(" Alpha:", alpha.Mul(0))
		}

		if m_sat.Mul(0) != 0 {
			LogIn(" Msat (A/m):", m_sat.Mul(0))
		} else {
			LogIn(" Msat (A/m): 0.0")
		}

		spins_val := calcSpins()

		if NSpins < 0 {
			errStr := "Panic Error: Number of spins must be greater than zero"
			LogErr(errStr)
			util.PanicErr(errors.New(errStr))
		} else {
			LogIn(" Num. spins:", spins_val)
		}

		LogIn(" GammaLL (rad/Ts):", GammaLL)

		if v.Mul(0) != 0 {
			LogIn(" Wc (rad/s):", v.Mul(0))
		}

		if uniform_vector.X() != 0.0 || uniform_vector.Y() != 0.0 || uniform_vector.Z() != 0.0 {
			LogIn(" Uniform vector (T): [", uniform_vector.X(), ",", uniform_vector.Y(), ",", uniform_vector.Z(), "]")
		}

		if c != nil {
			LogIn(" B_rms vector (T): [", getElemPos(c, X), ",", getElemPos(c, Y), ",", getElemPos(c, Z), "]")
		}

		if be != nil {
			LogIn(" B_ext vector (T): [", getElemPos(be, X), ",", getElemPos(be, Y), ",", getElemPos(be, Z), "]")
		}

		if FixDt != 0 {
			LogIn(" FixDt (s):", FixDt)
		}

		savePeriod := Table.autosave.period

		if savePeriod != 0 {
			LogIn(" Table autosave interval (s):", savePeriod)
		}

		if *simulationTime != 0 {
			LogIn(" Full simulation time (s):", *simulationTime)
		}

		LogIn("------------------------------------------------")
		LogIn("")
	}
}

func getElemPos(slice *data.Slice, position int) float32 {
	mz_temp := cuda.GetCell(slice, position, 0, 0, 0)
	return mz_temp
}

func calcFullSize() (float64, float64, float64, [3]float64, [3]int) {

	cell_size := Mesh().CellSize()
	size_cellx := cell_size[X]
	size_celly := cell_size[Y]
	size_cellz := cell_size[Z]

	num_cells := Mesh().Size()
	num_cellx := float64(num_cells[X])
	num_celly := float64(num_cells[Y])
	num_cellz := float64(num_cells[Z])

	full_sizex := size_cellx * num_cellx
	full_sizey := size_celly * num_celly
	full_sizez := size_cellz * num_cellz

	return full_sizex, full_sizey, full_sizez, cell_size, num_cells
}

// Calculate number of spins as a function of Msat
func calcSpins() float64 {

	if NSpins == 0 {

		m_sat := Msat.MSlice()
		defer m_sat.Recycle()

		full_sizex, full_sizey, full_sizez, _, _ := calcFullSize()

		full_vol := full_sizex * full_sizey * full_sizez

		NSpins = (full_vol * float64(m_sat.Mul(0))) / MuB
	}

	return NSpins
}

// Sets dst to the current total torque
func SetTorque(dst *data.Slice) {
	SetLLTorque(dst)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz torque
func SetLLTorque(dst *data.Slice) {

	SetEffectiveField(dst) // calculate and store B_eff

	alpha := Alpha.MSlice()
	defer alpha.Recycle()

	if Precess {
		cuda.LLTorque(dst, M.Buffer(), dst, alpha)
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}
}

func ApplyExtraFieldBeff(dst *data.Slice) {

	if !DisableTimeEvolutionTorque {

		sizeMesh := Mesh().Size()

		if mem_term.scn.Size() != sizeMesh {
			mem_term.Free()
		}

		if mem_term.scn == nil {
			mem_term.scn = cuda.NewSlice(MEMORY_COMPONENTS, sizeMesh)
			mem_term.last_time = 0.0
			mem_term.dt_time = 0.0
		}

		nspinsCalc := calcSpins()

		wc_slice := Wc.MSlice()
		defer wc_slice.Recycle()

		brms_slice := B_rms.MSlice()
		defer brms_slice.Recycle()

		mem_term.dt_time = Time - mem_term.last_time

		if !EnableCavityDissipation {
			// calculations without cavity dissipation
			cuda.SubSpinBextraBeff(dst, M.Buffer(), mem_term.scn, brms_slice, wc_slice, nspinsCalc, mem_term.dt_time, Time, GammaLL, Mesh())
		} else {

			// calculations with cavity dissipation
			kappa := Kappa.MSlice()
			defer kappa.Recycle()

			cuda.SubSpinBextraBeffDissipation(dst, M.Buffer(), mem_term.scn, brms_slice, wc_slice, kappa, nspinsCalc, mem_term.dt_time, Time, GammaLL, Mesh())
		}

		mem_term.last_time = Time
	}
}

// Adds the current spin transfer torque to dst
func AddSTTorque(dst *data.Slice) {
	if J.isZero() {
		return
	}
	util.AssertMsg(!Pol.isZero(), "spin polarization should not be 0")
	jspin, rec := J.Slice()
	if rec {
		defer cuda.Recycle(jspin)
	}
	fl, rec := FixedLayer.Slice()
	if rec {
		defer cuda.Recycle(fl)
	}
	if !DisableZhangLiTorque {
		msat := Msat.MSlice()
		defer msat.Recycle()
		j := J.MSlice()
		defer j.Recycle()
		alpha := Alpha.MSlice()
		defer alpha.Recycle()
		xi := Xi.MSlice()
		defer xi.Recycle()
		pol := Pol.MSlice()
		defer pol.Recycle()
		cuda.AddZhangLiTorque(dst, M.Buffer(), msat, j, alpha, xi, pol, GammaLL, Mesh())
	}
	if !DisableSlonczewskiTorque && !FixedLayer.isZero() {
		msat := Msat.MSlice()
		defer msat.Recycle()
		j := J.MSlice()
		defer j.Recycle()
		fixedP := FixedLayer.MSlice()
		defer fixedP.Recycle()
		alpha := Alpha.MSlice()
		defer alpha.Recycle()
		pol := Pol.MSlice()
		defer pol.Recycle()
		lambda := Lambda.MSlice()
		defer lambda.Recycle()
		epsPrime := EpsilonPrime.MSlice()
		defer epsPrime.Recycle()
		thickness := FreeLayerThickness.MSlice()
		defer thickness.Recycle()
		cuda.AddSlonczewskiTorque2(dst, M.Buffer(),
			msat, j, fixedP, alpha, pol, lambda, epsPrime,
			thickness,
			CurrentSignFromFixedLayerPosition[fixedLayerPosition],
			Mesh())
	}
}

func FreezeSpins(dst *data.Slice) {
	if !FrozenSpins.isZero() {
		cuda.ZeroMask(dst, FrozenSpins.gpuLUT1(), regions.Gpu())
	}
}

func (rk *MEMORY_TERM) Free() {
	rk.scn.Free()
	rk.scn = nil
	rk.last_time = 0.0
	rk.dt_time = 0.0
}

func GetMaxTorque() float64 {
	torque := ValueOf(Torque)
	defer cuda.Recycle(torque)
	return cuda.MaxVecNorm(torque)
}

type FixedLayerPosition int

const (
	FIXEDLAYER_TOP FixedLayerPosition = iota + 1
	FIXEDLAYER_BOTTOM
)

var (
	CurrentSignFromFixedLayerPosition = map[FixedLayerPosition]float64{
		FIXEDLAYER_TOP:    1.0,
		FIXEDLAYER_BOTTOM: -1.0,
	}
)

type flposition struct{}

func (*flposition) Eval() interface{}      { return fixedLayerPosition }
func (*flposition) SetValue(v interface{}) { drainOutput(); fixedLayerPosition = v.(FixedLayerPosition) }
func (*flposition) Type() reflect.Type     { return reflect.TypeOf(FixedLayerPosition(FIXEDLAYER_TOP)) }
