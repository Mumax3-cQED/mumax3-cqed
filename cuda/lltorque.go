package cuda

// MODIFIED INMA
import (
	"github.com/mumax/3/data"
)

var (
	sumx    *data.Slice = nil
	sumy    *data.Slice = nil
	sumz    *data.Slice = nil
	gt_dtsi int         = 0
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

	if timeEvolution == true {

		// if !IsBrmsZero(Brms_cuda) {

		// initBrmsSlice(m.Size())
		// initSumSlice(m.Size())

		size := m.Size()
		nx_size := size[X]
		ny_size := size[Y]
		nz_size := size[Z]

		if sumx == nil {
			sumx = NewSlice(1, size)
		}

		if sumy == nil {
			sumy = NewSlice(1, size)
		}

		if sumz == nil {
			sumz = NewSlice(1, size)
		}

		if Fixed_dt_cuda != 0.0 {
			gt_dtsi = 1
		} else {
			gt_dtsi = 0
		}

		ctimeWc := CurrentTime * Wc_cuda

		k_lltorque2time_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
			m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
			B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
			alpha.DevPtr(0), alpha.Mul(0),
			M_rk.DevPtr(0), M_rk.DevPtr(1), M_rk.DevPtr(2), M_rk.DevPtr(3), M_rk.DevPtr(4), M_rk.DevPtr(5), M_rk.DevPtr(6),
			M_rk.DevPtr(7), M_rk.DevPtr(8), M_rk.DevPtr(9),
			sumx.DevPtr(0), sumy.DevPtr(0), sumz.DevPtr(0),
			float32(ctimeWc), gt_dtsi, nx_size, ny_size, nz_size, N, cfg)

		// } else {
		// 	DefaultTorquePrecess(torque, m, B, alpha, N, cfg)
		// }

	} else {

		DefaultTorquePrecess(torque, m, B, alpha, N, cfg)
	}
}

func DefaultTorquePrecess(torque, m, B *data.Slice, alpha MSlice, N int, cfg *config) {

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
