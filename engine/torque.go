package engine

// MODIFIED INMA
import (
	"reflect"

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
	fixedLayerPosition                 = FIXEDLAYER_TOP // instructs mumax3 how free and fixed layers are stacked along +z direction

	B_rms  = NewExcitation("B_rms", "T", "Brms extra parameter for LLG time evolution")
	Wc     = NewScalarParam("Wc", "rad/s", "Wc extra parameter for LLG time evolution")
	NSpins = NewScalarParam("NSpins", "", "Number of spins")

	Bext_custom float64 = 0.0

	scn    *data.Slice
	deltah float64 = 0.0
	ctime  float64 = 0.0
	dtw    float64 = 0.0
)

func init() {
	Pol.setUniform([]float64{1}) // default spin polarization
	Lambda.Set(1)                // sensible default value (?).
	DeclVar("GammaLL", &GammaLL, "Gyromagnetic ratio in rad/Ts")
	DeclVar("Bext_custom", &Bext_custom, "B external field custom in T")
	DeclVar("DisableZhangLiTorque", &DisableZhangLiTorque, "Disables Zhang-Li torque (default=false)")
	DeclVar("DisableSlonczewskiTorque", &DisableSlonczewskiTorque, "Disables Slonczewski torque (default=false)")
	DeclVar("DisableTimeEvolutionTorque", &DisableTimeEvolutionTorque, "Disables Time evolution torque (default=true)")
	DeclVar("DisableBeffContributions", &DisableBeffContributions, "Disables Beff default contributions (default=false)")
	DeclVar("DoPrecess", &Precess, "Enables LL precession (default=true)")
	DeclLValue("FixedLayerPosition", &flposition{}, "Position of the fixed layer: FIXEDLAYER_TOP, FIXEDLAYER_BOTTOM (default=FIXEDLAYER_TOP)")
	DeclROnly("FIXEDLAYER_TOP", FIXEDLAYER_TOP, "FixedLayerPosition = FIXEDLAYER_TOP instructs mumax3 that fixed layer is on top of the free layer")
	DeclROnly("FIXEDLAYER_BOTTOM", FIXEDLAYER_BOTTOM, "FixedLayerPosition = FIXEDLAYER_BOTTOM instructs mumax3 that fixed layer is underneath of the free layer")
}

func PrintParametersTimeEvolution() {

	if !DisableTimeEvolutionTorque {

		c, _ := B_rms.Slice()
		be, _ := B_ext.Slice()
		v := Wc.MSlice()
		ns := NSpins.MSlice()
		m_sat := Msat.MSlice()
		alpha := Alpha.MSlice()

		LogIn("")
		LogIn("------------------------------------------------")
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

		cell_size := Mesh().CellSize()
		num_cells := Mesh().Size()

		LogIn(" Cell size (m):", cell_size[X], "x", cell_size[Y], "x", cell_size[Z])
		LogIn(" Num. cells:", num_cells[X], "x", num_cells[Y], "x", num_cells[Z])
		LogIn(" Alpha:", alpha.Mul(0))
		LogIn(" B_ext custom (T):", Bext_custom)
		LogIn(" Num. spins:", ns.Mul(0))

		if m_sat.Mul(0) != 0.0 {
			LogIn(" Msat (A/m):", m_sat.Mul(0))
		} else {
			LogIn(" Msat (A/m): 0.0")
		}

		LogIn(" GammaLL (rad/Ts):", GammaLL)
		LogIn(" Wc (rad/s):", v.Mul(0))
		LogIn(" Brms vector (T): [", cuda.GetElemPos(c, X), cuda.GetElemPos(c, Y), cuda.GetElemPos(c, Z), "]")
		LogIn(" B_ext vector (T): [", cuda.GetElemPos(be, X), cuda.GetElemPos(be, Y), cuda.GetElemPos(be, Z), "]")

		if FixDt != 0 {
			LogIn(" FixDt (s):", FixDt)
		}

		LogIn("------------------------------------------------")
		LogIn("")

		defer c.Free()
		defer ns.Recycle()
		defer be.Free()
		defer v.Recycle()
		defer m_sat.Recycle()
		defer alpha.Recycle()
	}
}

// Sets dst to the current total torque
func SetTorque(dst *data.Slice) {
	SetLLTorque(dst)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz torque
func SetLLTorque(dst *data.Slice) {

	if !DisableTimeEvolutionTorque {
		SetTempValues(Time, Dt_si, Dt_Weighted)
	}

	SetEffectiveField(dst) // calc and store B_eff

	alpha := Alpha.MSlice()
	defer alpha.Recycle()

	if Precess {
		cuda.LLTorque(dst, M.Buffer(), dst, alpha)
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}
}

func getScnSlice() *data.Slice {

	if scn == nil {
		scn = cuda.NewSlice(2, M.Buffer().Size())
	}

	return scn
}

func ApplyExtraFieldBeff(dst *data.Slice) {

	if !DisableTimeEvolutionTorque {

		wc_slice := Wc.MSlice()
		defer wc_slice.Recycle()

		brms_slice := B_rms.MSlice()
		defer brms_slice.Recycle()

		nspins := NSpins.MSlice()
		defer nspins.Recycle()

		cuda.SubSpinBextraBeff(dst, M.Buffer(), getScnSlice(), wc_slice, brms_slice, nspins, ctime, dtw, Mesh())
		dtw = 0.0
		Dt_Weighted = 0.0
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
		cuda.AddZhangLiTorque(dst, M.Buffer(), msat, j, alpha, xi, pol, Mesh())
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

// New function for LLG formula time evolution
func SetTempValues(time, delta, dtweight float64) {
	ctime = time
	deltah = delta
	dtw = dtweight
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
