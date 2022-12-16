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
	// func LLTorque(torque, m, B *data.Slice, alpha MSlice, hbar_factor int) {
	N := torque.Len()
	cfg := make1DConf(N)

	k_lltorque2_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		alpha.DevPtr(0), alpha.Mul(0), N, cfg)
	// alpha.DevPtr(0), alpha.Mul(0), hbar_factor, N, cfg)
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
func LLTimeTorque(torque, new_term *data.Slice) {

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
	// if InternalTimeLatch {

	N := torque.Len()
	cfg := make1DConf(N)
	// fmt.Println(GetElemPos(new_term, 2))
	//	fmt.Println(GetZElem(New_term_llg))
	k_lltorque2time_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		new_term.DevPtr(X),
		new_term.DevPtr(Y),
		new_term.DevPtr(Z),
		N, cfg)

	// fmt.Println(GetElemPos(torque, 0))
	// }
	// 	} else {
	// 		DefaultTorquePrecess(torque, m, B, alpha, N, cfg)
	// 	}
	//
	// } else {
	//
	// 	DefaultTorquePrecess(torque, m, B, alpha, N, cfg)
	// }
}

func CalcTempTorque(dst_slice, m, sin_sum, cos_sum, sum_slice *data.Slice, layer, wc, brms, alpha MSlice, ctime float64, deltah float32) {

	N := dst_slice.Len()
	cfg := make1DConf(N)

	k_mdatatemp_async(dst_slice.DevPtr(X), dst_slice.DevPtr(Y), dst_slice.DevPtr(Z),
		sin_sum.DevPtr(X), sin_sum.DevPtr(Y), sin_sum.DevPtr(Z),
		cos_sum.DevPtr(X), cos_sum.DevPtr(Y), cos_sum.DevPtr(Z),
		sum_slice.DevPtr(X), sum_slice.DevPtr(Y), sum_slice.DevPtr(Z),
		layer.DevPtr(X), layer.Mul(X),
		layer.DevPtr(Y), layer.Mul(Y),
		layer.DevPtr(Z), layer.Mul(Z),
		wc.DevPtr(0), wc.Mul(0),
		brms.DevPtr(X), brms.Mul(X),
		brms.DevPtr(Y), brms.Mul(Y),
		brms.DevPtr(Z), brms.Mul(Z),
		alpha.DevPtr(0), alpha.Mul(0),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		deltah, float32(ctime), N, cfg)

	// fmt.Println(GetElemPos(wc.arr, 0))
	// fmt.Println("0:", GetElemPos(dst_slice, 0))
	// fmt.Println("1:", GetElemPos(dst_slice, 1))
	// fmt.Println("2:", GetElemPos(dst_slice, 2))
}
