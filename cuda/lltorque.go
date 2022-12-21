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

// Apply new value Spin Torque to Beff
func CalcSpinTorque(dst_slice, m, sin_sum, cos_sum *data.Slice, msat, wc, brms MSlice, ctime float64, deltah float32, mesh *data.Mesh) {

	N := mesh.Size()
	cfg := make3DConf(N)
	pbc := mesh.PBC_code()
	c := mesh.CellSize()
	vol := c[X] * c[Y] * c[Z]

	k_addspin2beff_async(dst_slice.DevPtr(X), dst_slice.DevPtr(Y), dst_slice.DevPtr(Z),
		sin_sum.DevPtr(X), sin_sum.DevPtr(Y), sin_sum.DevPtr(Z),
		cos_sum.DevPtr(X), cos_sum.DevPtr(Y), cos_sum.DevPtr(Z),
		wc.DevPtr(0), wc.Mul(0), msat.DevPtr(0), msat.Mul(0),
		brms.DevPtr(X), brms.Mul(X),
		brms.DevPtr(Y), brms.Mul(Y),
		brms.DevPtr(Z), brms.Mul(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		deltah, float32(ctime), float32(vol), N[X], N[Y], N[Z], pbc, cfg)

	//fmt.Println(GetElemPos(wc.arr, 0))
	//fmt.Println("deltat:", deltah/float32(1.7595e11))
	//fmt.Println("0:", GetElemPos(dst_slice, 0))
	//fmt.Println("1:", GetElemPos(dst_slice, 1))
	//fmt.Println("2:", GetElemPos(dst_slice, 2))
}
