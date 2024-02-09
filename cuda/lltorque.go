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

// Apply new value Spin Torque to Beff --> Beff - Bcustom with cavity dissipation
func SubSpinBextraBeff(dst, m, scn *data.Slice, brms, wc, kappa MSlice, m0, p0, nspins, deltah, ctime, gammaLL float64, mesh *data.Mesh) {

	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)

	k_calcspinbeff_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		scn.DevPtr(0), scn.DevPtr(1),
		wc.DevPtr(0), wc.Mul(0),
		kappa.DevPtr(0), kappa.Mul(0),
		brms.DevPtr(X), brms.Mul(X),
		brms.DevPtr(Y), brms.Mul(Y),
		brms.DevPtr(Z), brms.Mul(Z),
		float32(m0), float32(p0), float32(nspins), float32(deltah), float32(ctime), float32(gammaLL), N[X], N[Y], N[Z], pbc, cfg)
}
