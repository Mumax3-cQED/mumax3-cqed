package engine

// MODIFIED INMA
import (
	"reflect"
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

	B_rms                        = NewExcitation("B_rms", "T", "Zero point magnetic field of the cavity")
	Wc                           = NewScalarParam("Wc", "rad/s", "Resonant frequency of the cavity")
	Kappa                        = NewScalarParam("Kappa", "rad/s", "Cavity dissipation")
	NSpins          float64      = 0          // Number of spins
	StartCheckpoint time.Time    = time.Now() // Starting date for mumax3 script to measure elapsed execution time, to set starting date anywhere in the  --> StartCheckpoint = now()
	mem_term        *MEMORY_TERM = nil
)

const (
	MuB               = 9.2740091523E-24
	MEMORY_COMPONENTS = 2
)

// Equation Memory Term
type MEMORY_TERM struct {
	scn       *data.Slice
	last_time float64
	dt_time   float64
}

func init() {

	mem_term = new(MEMORY_TERM)  // init new memory term for equation
	Pol.setUniform([]float64{1}) // default spin polarization
	Lambda.Set(1)                // sensible default value (?).

	DeclVar("StartCheckpoint", &StartCheckpoint, "Script launch starting date (default now() at the beginning of mumax3 allocation)")
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

// Compute new extra term in effective field (see effectivefield.go)
func ApplyExtraFieldBeff(dst *data.Slice) {

	sizeMesh := Mesh().Size()

	if mem_term.scn != nil && mem_term.scn.Size() != sizeMesh {
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

// Free memory resources
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
