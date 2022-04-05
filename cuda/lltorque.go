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

	// k_lltorque2_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
	// 	m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
	// 	B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
	// 	alpha.DevPtr(0), alpha.Mul(0), N, util.Bextra_vector, cfg)

	// log.Println("entra CUDA")
	if TimeEvo == true {

		if LockMExec {

			// mx_temp := GetCell(m, 0, 0, 0, 0)
			// my_temp := GetCell(m, 1, 0, 0, 0)
			// mz_temp := GetCell(m, 2, 0, 0, 0)
			//
			// log.Println("vvvv2:", mx_temp, ", ", my_temp, ", ", mz_temp)

			// log.Println("entra true aqui")
			// log.Println("entra time evo")
			initBrmsSlice(m.Size())

			var hbar_constant float32 = 2 / 1.054571817e-34

			k_lltorque2time_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
				m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
				B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
				alpha.DevPtr(0), alpha.Mul(0), float32(Fixed_dt_cuda), float32(Wc_cuda), float32(Brms_cuda[X]), float32(Brms_cuda[Y]), float32(Brms_cuda[Z]),
				Brms_i.DevPtr(0), Brms_i.DevPtr(1), Brms_i.DevPtr(2),
				M_rk.DevPtr(0), M_rk.DevPtr(1), M_rk.DevPtr(2), M_rk.DevPtr(3), M_rk.DevPtr(4), M_rk.DevPtr(5), M_rk.DevPtr(6), hbar_constant, N, cfg)

		} else {
			DefaultTorquePrecess(torque, m, B, alpha, N, cfg)
		}

	} else {
		DefaultTorquePrecess(torque, m, B, alpha, N, cfg)
	}
}

func DefaultTorquePrecess(torque, m, B *data.Slice, alpha MSlice, N int, cfg *config) {

	// debug.PrintStack()

	// log.Println("entra time no evo")
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
