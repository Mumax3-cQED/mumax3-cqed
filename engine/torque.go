package engine

// MODIFIED INMA
import (
	"reflect"
	 // "math"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	  // "fmt"
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
	J                                  = NewExcitation("J", "A/m2", "Electrical current density")
	MaxTorque                          = NewScalarValue("maxTorque", "T", "Motion term for LLG equation", GetMaxTorque)
	GammaLL                    float64 = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	Precess                            = true
	DisableZhangLiTorque               = false
	DisableSlonczewskiTorque           = false
	DisableTimeEvolutionTorque         = true
	fixedLayerPosition                 = FIXEDLAYER_TOP // instructs mumax3 how free and fixed layers are stacked along +z direction
	Brms_vector                [3]float64
	Wc                         float64 = 0.0
	MTTorqueFixedLayer                 = NewExcitation("MTTorqueFixedLayer", "", "MTTorqueFixedLayer fixed layer")
	// cosine_sum                         = NewExcitation("cosine_sum", "Hz", "Cosine sum")
	// sine_sum                           = NewExcitation("sine_sum", "Hz", "Sine sum")
	// result_sum                         = NewExcitation("result_sum", "Hz", "Full sum result")
	// dst_res 													 = NewExcitation("dst_res", "Hz", "Full new result")
	MTTorque                           = NewVectorField("MTTorque", "T", "Spin-transfer torque/γ0", AddLLTimeTorque)

	resultsum_slice, sum_slice, sin_slice, cos_slice, dst_slice *data.Slice
	layer_slice cuda.MSlice
	//cosine_sum, sine_sum, full_sum,
	// dst_res *data.Slice
	// sin_slice_test, sin_wctime_slice *data.Slice
	// result_slice, sum_slice, sin_slice, cos_slice *data.Slice
	// result_sin_slice, result_op, result_cos_slice, result_sub, cos_wctime_slice, sin_wctime_slice *data.Slice
	 HBAR float64 = 1.05457173E-34
)

func init() {
	Pol.setUniform([]float64{1}) // default spin polarization
	Lambda.Set(1)                // sensible default value (?).
	DeclVar("GammaLL", &GammaLL, "Gyromagnetic ratio in rad/Ts")
	DeclVar("DisableZhangLiTorque", &DisableZhangLiTorque, "Disables Zhang-Li torque (default=false)")
	DeclVar("DisableSlonczewskiTorque", &DisableSlonczewskiTorque, "Disables Slonczewski torque (default=false)")
	DeclVar("DisableTimeEvolutionTorque", &DisableTimeEvolutionTorque, "Disables Time evolution torque (default=true)")
	DeclVar("B_rms", &Brms_vector, "Brms extra parameter for LLG time evolution")
	DeclVar("Wc", &Wc, "Wc extra parameter for LLG time evolution")
	DeclVar("DoPrecess", &Precess, "Enables LL precession (default=true)")
	DeclLValue("FixedLayerPosition", &flposition{}, "Position of the fixed layer: FIXEDLAYER_TOP, FIXEDLAYER_BOTTOM (default=FIXEDLAYER_TOP)")
	DeclROnly("FIXEDLAYER_TOP", FIXEDLAYER_TOP, "FixedLayerPosition = FIXEDLAYER_TOP instructs mumax3 that fixed layer is on top of the free layer")
	DeclROnly("FIXEDLAYER_BOTTOM", FIXEDLAYER_BOTTOM, "FixedLayerPosition = FIXEDLAYER_BOTTOM instructs mumax3 that fixed layer is underneath of the free layer")
}

// Sets dst to the current total torque
func SetTorque(dst *data.Slice) {
	SetLLTorque(dst)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

func SetTorqueTime(dst *data.Slice) {
	SetLLTorque(dst)
	AddLLTimeTorque(dst)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz torque
func SetLLTorque(dst *data.Slice) {
	SetEffectiveField(dst) // calc and store B_eff
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	if Precess {
		cuda.LLTorque(dst, M.Buffer(), dst, alpha) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
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
		cuda.AddSlonczewskiTorque2(dst, M.Buffer(),
			msat, j, fixedP, alpha, pol, lambda, epsPrime,
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
func ComputeNewTerm(ctime float64, deltah float32) {

	 // m_current := M.Buffer()
   // size := m_current.Size()

	// if sin_slice_test.IsNil() {
	// 	sin_slice_test = cuda.NewSlice(3, size)
	// }
	//
	// if cos_slice.IsNil() {
	// 	cos_slice = cuda.NewSlice(3, size)
	// }
	//
	// if sum_slice.IsNil() {
	// 	sum_slice =cuda.NewSlice(3, size)
	// }
	//
	// if result_slice.IsNil() {
	// 	result_slice = cuda.NewSlice(3, size)
	// }
	//
	// if result_sin_slice.IsNil() {
	// 	result_sin_slice = cuda.NewSlice(3, size)
	// }
	//
	// if result_cos_slice.IsNil() {
	// 	result_cos_slice = cuda.NewSlice(3, size)
	// }
	//
	// if sin_wctime_slice.IsNil() {
	// 	sin_wctime_slice = cuda.NewSlice(3, size)
	// }
	//
	// if cos_wctime_slice.IsNil() {
	// 	cos_wctime_slice = cuda.NewSlice(3, size)
	// }
	//
	// if result_sub.IsNil() {
	// 	result_sub = cuda.NewSlice(3, size)
	// }
	// if result_op.IsNil() {
	// 	result_op = cuda.NewSlice(3, size)
	// }
	//
	// sin_wctime_slice.SetScalar(size[X], size[Y], size[Z], math.Sin(Wc * ctime))
	//
	// cuda.Mul(sin_wctime_slice, m, sin_wctime_slice)
	// cuda.Add(sin_slice_test, sin_slice_test, sin_wctime_slice)
	//
	// cos_wctime_slice.SetScalar(size[X], size[Y], size[Z], math.Cos(Wc * ctime))
	//
	// cuda.Mul(cos_wctime_slice, m, cos_wctime_slice)
	// cuda.Add(cos_slice, cos_slice, cos_wctime_slice)
	//
	// cuda.Mul(result_sin_slice, sin_slice, cos_wctime_slice)
	// cuda.Mul(result_cos_slice, cos_slice, sin_wctime_slice)
	//
	// cuda.Msub2(result_sub, result_sin_slice, result_cos_slice, 1, 1)
	//
	// deltah_slice := cuda.NewSlice(3, size)
	// deltah_slice.SetScalar(size[X], size[Y], size[Z],float64(deltah))
	//
	// brms_slice := cuda.NewSlice(3, size)
	// brms_slice.SetVector(size[X], size[Y], size[Z], Brms_vector)
	//
	// constant_slice := cuda.NewSlice(3, size)
	//
	// cuda.Mul(constant_slice, brms_slice, deltah_slice)
	//
	// result_sum_slice := cuda.NewSlice(3, size)
	// cuda.Mul(result_sum_slice, constant_slice, result_sub)
	//
	// cuda.Mul(sum_slice, sum_slice, result_sum_slice)
	//
	// result_mxbrms_slice := cuda.NewSlice(3, size)
	// cuda.CrossProduct(result_mxbrms_slice, m, brms_slice)
	//
	// spin_constant  := 2 / HBAR
	//
	// result_op.SetScalar(size[X], size[Y], size[Z], spin_constant)
	//
	// cuda.Mul(result_op, result_op, result_mxbrms_slice)
	// cuda.Mul(result_op, result_op, sum_slice)
size := M.Buffer().Size()
	if dst_slice.DevPtr(0) == nil {
		 // dst_slice, _ = dst_res.Slice()
		  dst_slice = cuda.NewSlice(3, size)
	}

	if sin_slice.DevPtr(0) == nil {
		// sin_slice, _ = sine_sum.Slice()
		sin_slice = cuda.NewSlice(3, size)
	}

	if cos_slice.DevPtr(0) == nil {
		// cos_slice, _ = cosine_sum.Slice()
		cos_slice = cuda.NewSlice(3, size)
	}

	if resultsum_slice.DevPtr(0) == nil {
		// resultsum_slice, _ = result_sum.Slice()
		resultsum_slice = cuda.NewSlice(3, size)
	}

	if layer_slice.DevPtr(0) == nil {
		layer_slice = MTTorqueFixedLayer.MSlice()
	}

	// alpha := Alpha.MSlice()
	// defer alpha.Recycle()

	cuda.CalcTempTorque(sin_slice, cos_slice, resultsum_slice, dst_slice, M.Buffer(), layer_slice, Wc, ctime, deltah, Brms_vector)
	cuda.Normalize(dst_slice, geometry.Gpu())
}

func AddLLTimeTorque(dst *data.Slice) {

	if !DisableTimeEvolutionTorque {

		// if result_op.IsNil() {
		// 	return
		// }
		if dst_slice.IsNil() {
			return
		}

      //cuda.Msub2(dst, dst, dst_slice, 1, 1)

		    // dst_res.SubTo(dst)

		       cuda.LLTimeTorque(dst, dst_slice)

	}
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
