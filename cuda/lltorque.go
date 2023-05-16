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

// Apply new value Spin Torque to Beff --> Beff - Bcustom
func SubSpinBextraBeff(dst, m, scn *data.Slice, brms, wc, nspins MSlice, ctime, deltah, gammaLL float64, mesh *data.Mesh) {
	//N := dst.Len()
	N := mesh.Size()
	// util.Assert(m.Size() == N)
	cfg := make3DConf(N)
	pbc := mesh.PBC_code()
	//c := mesh.CellSize()
	//vol := float32(c[X]*c[Y]*c[Z]) / float32(N[X]*N[Y]*N[Z])
	// fmt.Println("aqui")
	k_calcspinbeff_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		m.DevPtr(X),
		m.DevPtr(Y),
		m.DevPtr(Z),
		scn.DevPtr(X), scn.DevPtr(Y),
		wc.DevPtr(0), wc.Mul(0),
		nspins.DevPtr(0), nspins.Mul(0),
		brms.DevPtr(X), brms.Mul(X),
		brms.DevPtr(Y), brms.Mul(Y),
		brms.DevPtr(Z), brms.Mul(Z),
		float32(deltah), float32(ctime), float32(gammaLL), N[X], N[Y], N[Z], pbc, cfg)
	// deltah, ctime, float32(vol), N[X], N[Y], N[Z], pbc, cfg)

}
