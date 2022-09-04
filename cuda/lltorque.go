package cuda

// MODIFIED INMA
import (
	"github.com/mumax/3/data"
)

// Landau-Lifshitz torque divided by gamma0:
// 	- 1/(1+α²) [ m x B +  α m x (m x B) ]
// 	torque in Tesla
// 	m normalized
// 	B in Tesla
// see lltorque.cu
func LLTorque(torque, m, B *data.Slice, alpha MSlice) {
	N := torque.Len()
	cfg := make1DConf(N)

	k_lltorque2_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		alpha.DevPtr(0), alpha.Mul(0), N, cfg)
}

// Landau-Lifshitz torque with precession disabled.
// Used by engine.Relax().
func LLNoPrecess(torque, m, B *data.Slice) {
	N := torque.Len()
	cfg := make1DConf(N)

	k_llnoprecess_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z), N, cfg)
}

// func LLTimeTorque(torque, m, B *data.Slice, alpha MSlice, mesh *data.Mesh) {
func LLTimeTorque(torque *data.Slice) {

	// if timeEvolution == true {

	// if !IsBrmsZero(Brms_cuda) {

	// initBrmsSlice(m.Size())
	// initSumSlice(m.Size())
	// size := N
	// if sumx == nil {
	//
	// 	sumx = NewSlice(1, size)
	// }
	//
	// if sumy == nil {
	// 	sumy = NewSlice(1, size)
	// }
	//
	// if sumz == nil {
	// 	sumz = NewSlice(1, size)
	// }

	// if Fixed_dt_cuda != 0.0 {
	// 	gt_dtsi = 1
	//
	// } else {
	// 	gt_dtsi = 0
	//
	// }
	//
	// ctimeWc := CurrentTime * Wc_cuda
	//
	// k_lltorque2time_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
	// 	m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
	// 	B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
	// 	alpha.DevPtr(0), alpha.Mul(0),
	// 	M_rk.DevPtr(0), M_rk.DevPtr(1), M_rk.DevPtr(2), M_rk.DevPtr(3), M_rk.DevPtr(4), M_rk.DevPtr(5), M_rk.DevPtr(6),
	// 	M_rk.DevPtr(7), M_rk.DevPtr(8), M_rk.DevPtr(9),
	// 	sumx.DevPtr(0), sumy.DevPtr(0), sumz.DevPtr(0),
	// 	float32(ctimeWc), gt_dtsi, N[X], N[Y], N[Z], cfg)
	if InternalTimeLatch {

		N := torque.Len()
		cfg := make1DConf(N)

		k_lltorque2time_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
			New_term_llg.DevPtr(0), New_term_llg.DevPtr(1), New_term_llg.DevPtr(2), N, cfg)
	}
	// 	} else {
	// 		DefaultTorquePrecess(torque, m, B, alpha, N, cfg)
	// 	}
	//
	// } else {
	//
	// 	DefaultTorquePrecess(torque, m, B, alpha, N, cfg)
	// }
}

func CalcMSpinTorque(dst, m_current *data.Slice, ctime float64, deltah float32, brms [3]float64, wc float64) {

	N := m_current.Len()
	cfg := make1DConf(N)

	k_mdatatemp_async(dst.DevPtr(0), dst.DevPtr(1), dst.DevPtr(2), dst.DevPtr(3), dst.DevPtr(4), dst.DevPtr(5), dst.DevPtr(6), dst.DevPtr(7), dst.DevPtr(8), dst.DevPtr(9),
		m_current.DevPtr(0), m_current.DevPtr(1), m_current.DevPtr(2),
		float32(wc*ctime), float32(deltah), float32(brms[0]), float32(brms[1]), float32(brms[2]), N, cfg)
}

func CalcStepNewTerm(dst, data, sum_temp, m_current *data.Slice, ctime float64, wc float64) {

	N := m_current.Len()
	cfg := make1DConf(N)

	k_term2time_async(dst.DevPtr(0), dst.DevPtr(1), dst.DevPtr(2),
		data.DevPtr(0), data.DevPtr(1), data.DevPtr(2), data.DevPtr(3), data.DevPtr(4), data.DevPtr(5), data.DevPtr(6), data.DevPtr(7), data.DevPtr(8), data.DevPtr(9),
		m_current.DevPtr(0), m_current.DevPtr(1), m_current.DevPtr(2),
		sum_temp.DevPtr(0), sum_temp.DevPtr(1), sum_temp.DevPtr(2),
		float32(wc*ctime), N, cfg)
}
