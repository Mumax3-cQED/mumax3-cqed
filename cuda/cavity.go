package cuda

// Add cavity field to Beff
import (
	"math"

	"github.com/mumax/3/data"
)

// Add exchange field to Beff.
// 	m: normalized magnetization
//  brms: Zero point magnetic field in Tesla
//  wc: Cavity frequency in rad/s-1
// 	kappa: Cavity dissipation in rad/s-1
//  x0: Cavity initial condtion
//  p0: Cavity initial condtion
// see cavity.cu
func AddCavity(dst, full_m, brms, scn *data.Slice, wc, kappa MSlice, x0, p0, msatCell, dt, ctime float64, mem *[2]float64, mesh *data.Mesh, customKernel bool) {

	N := mesh.Size()
	brms = data.Resample(brms, N) // reshape of OVF Brms file to mesh size
	brms_m := Dot(brms, full_m)

	if customKernel {
		cfg := make3DConf(N)
		pbc := mesh.PBC_code()

		k_addcavity_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
			scn.DevPtr(0), scn.DevPtr(1),
			wc.DevPtr(0), wc.Mul(0),
			kappa.DevPtr(0), kappa.Mul(0),
			brms.DevPtr(X),
			brms.DevPtr(Y),
			brms.DevPtr(Z),
			float32(x0), float32(p0), float32(msatCell), float32(dt), float32(ctime), brms_m, N[X], N[Y], N[Z], pbc, cfg)
	} else {

		kappa_temp := float64(kappa.Mul(0))
		wc_temp := float64(wc.Mul(0))

		(*mem)[0] += math.Exp(kappa_temp*ctime) * math.Sin(wc_temp*ctime) * float64(brms_m) * dt
		(*mem)[1] += math.Exp(kappa_temp*ctime) * math.Cos(wc_temp*ctime) * float64(brms_m) * dt

		G := math.Exp(-kappa_temp*ctime) * (math.Cos(wc_temp*ctime)*(x0-msatCell*(*mem)[0]) - math.Sin(wc_temp*ctime)*(p0-msatCell*(*mem)[1]))

		Madd2(dst, dst, brms, 1.0, float32(G))
	}
}
