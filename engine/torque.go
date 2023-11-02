package engine

// MODIFIED INMA
import (
	"errors"
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
	EnableCavityDissipation            = false
	fixedLayerPosition                 = FIXEDLAYER_TOP // instructs mumax3 how free and fixed layers are stacked along +z direction

	B_rms  = NewExcitation("B_rms", "T", "Zero point magnetic field of the cavity")
	Wc     = NewScalarParam("Wc", "rad/s", "Resonant frequency of the cavity")
	Ws     = NewScalarParam("Ws", "rad/s", "Resonant frequency of the spins")
	Kappa  = NewScalarParam("Kappa", "rad/s", "Cavity dissipation")
	NSpins = NewScalarParam("NSpins", "", "Number of spins")

	mem_term *MEMORY_TERM = nil
)

const (
	MuB         = 9.2740091523E-24
	MEMORY_COMP = 6
)

// Memory term computation
type MEMORY_TERM struct {
	scn       *data.Slice
	last_time float64
}

func init() {
	mem_term = new(MEMORY_TERM)
	Pol.setUniform([]float64{1}) // default spin polarization
	Lambda.Set(1)                // sensible default value (?).
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
}

func PrintParametersTimeEvolution(simulationTime *float64) {

	if !DisableTimeEvolutionTorque {

		// check if not empty
		if mem_term.scn != nil {
			mem_term.Free()
		}

		c, r1 := B_rms.Slice()
		if r1 {
			defer cuda.Recycle(c)
		}
		//	println(c.Size()[0], " ", c.Size()[1], " ", c.Size()[2])
		be, r2 := B_ext.Slice()
		if r2 {
			defer cuda.Recycle(be)
		}

		v := Wc.MSlice()
		defer v.Recycle()

		ws_info := Ws.MSlice()
		defer ws_info.Recycle()

		ns := NSpins.MSlice()
		defer ns.Recycle()

		m_sat := Msat.MSlice()
		defer m_sat.Recycle()

		alpha := Alpha.MSlice()
		defer alpha.Recycle()

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

		if EnableCavityDissipation {
			LogIn(" Cavity Dissipation: Enabled")

			kappa := Kappa.MSlice()
			defer kappa.Recycle()

			LogIn(" Kappa (rad/s):", kappa.Mul(0))
		} else {
			LogIn(" Cavity Dissipation: Disabled")
		}

		cell_size := Mesh().CellSize()
		num_cells := Mesh().Size()

		LogIn(" Shape size (m):", cell_size[X]*float64(num_cells[X]), "x", cell_size[Y]*float64(num_cells[Y]), "x", cell_size[Z]*float64(num_cells[Z]))
		LogIn(" Num. cells:", num_cells[X], "x", num_cells[Y], "x", num_cells[Z])
		LogIn(" Cell size (m):", cell_size[X], "x", cell_size[Y], "x", cell_size[Z])
		LogIn(" Alpha:", alpha.Mul(0))

		if m_sat.Mul(0) != 0.0 {
			LogIn(" Msat (A/m):", m_sat.Mul(0))
		} else {
			LogIn(" Msat (A/m): 0.0")
		}

		if ns.Mul(0) < 0.0 {

			errStr := "Panic Error: Number of spins must be greater than zero"
			LogErr(errStr)
			util.PanicErr(errors.New(errStr))

		} else if ns.Mul(0) == 0.0 {

			calcSpins()

			ns_upd := NSpins.MSlice()
			defer ns_upd.Recycle()

			LogIn(" Num. spins:", ns_upd.Mul(0))

		} else {
			LogIn(" Num. spins:", ns.Mul(0))
		}

		LogIn(" GammaLL (rad/Ts):", GammaLL)

		if v.Mul(0) != 0.0 {
			LogIn(" Wc (rad/s):", v.Mul(0))
		}

		if ws_info.Mul(0) != 0.0 {
			LogIn(" Ws (rad/s):", ws_info.Mul(0))
		}

		LogIn(" B_rms vector (T): [", cuda.GetElemPos(c, X), ",", cuda.GetElemPos(c, Y), ",", cuda.GetElemPos(c, Z), "]")
		LogIn(" B_ext vector (T): [", cuda.GetElemPos(be, X), ",", cuda.GetElemPos(be, Y), ",", cuda.GetElemPos(be, Z), "]")

		if FixDt != 0 {
			LogIn(" FixDt (s):", FixDt)
		}

		LogIn(" Full simulation time (s):", *simulationTime)
		LogIn("------------------------------------------------")
		LogIn("")
	}
}

// Calculate number of spins as function of Msat and set to NSpins variable
func calcSpins() {

	ns := NSpins.MSlice()
	defer ns.Recycle()

	if ns.Mul(0) == 0.0 {

		m_sat := Msat.MSlice()
		defer m_sat.Recycle()

		cell_size := Mesh().CellSize()

		size_cellx := cell_size[X]
		size_celly := cell_size[Y]
		size_cellz := cell_size[Z]

		nspins_calc := (size_cellx * size_celly * size_cellz * float64(m_sat.Mul(0))) / MuB

		NSpins.Set(nspins_calc)
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

		if mem_term.scn.Size() != Mesh().Size() {
			mem_term.Free()
		}

		if mem_term.scn == nil {
			mem_term.scn = cuda.NewSlice(MEMORY_COMP, Mesh().Size())
		}

		calcSpins()

		wc_slice := Wc.MSlice()
		defer wc_slice.Recycle()

		brms_slice := B_rms.MSlice()
		defer brms_slice.Recycle()

		nspins := NSpins.MSlice()
		defer nspins.Recycle()

		dt_time := Time - mem_term.last_time

		if !EnableCavityDissipation {
			// calculations without cavity dissipation
			cuda.SubSpinBextraBeff(dst, M.Buffer(), mem_term.scn, brms_slice, wc_slice, nspins, dt_time, Time, GammaLL, Mesh())
		} else {

			// calculations with cavity dissipation
			kappa := Kappa.MSlice()
			defer kappa.Recycle()

			cuda.SubSpinBextraBeffDissipation(dst, M.Buffer(), mem_term.scn, brms_slice, wc_slice, nspins, kappa, dt_time, Time, GammaLL, Mesh())
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
