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
func AddCavity(dst, m, brms *data.Slice, wc, kappa MSlice, x0, p0, msatCell, deltah, ctime float64, mem *[2]float64, mesh *data.Mesh, customKernel bool) {

	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	brms = data.Resample(brms, N) // reshape of OVF Brms file to mesh size
	brms_m := float64(Dot(brms, m))

	kappa_temp := float64(kappa.Mul(0))
	wc_temp := float64(wc.Mul(0))

	(*mem)[0] += math.Exp(kappa_temp*ctime) * math.Sin(wc_temp*ctime) * brms_m * deltah
	(*mem)[1] += math.Exp(kappa_temp*ctime) * math.Cos(wc_temp*ctime) * brms_m * deltah

	if customKernel {
		k_addcavity_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
			wc.DevPtr(0), wc.Mul(0),
			kappa.DevPtr(0), kappa.Mul(0),
			brms.DevPtr(X),
			brms.DevPtr(Y),
			brms.DevPtr(Z),
			float32(x0), float32(p0), float32(msatCell), float32(ctime), float32((*mem)[0]), float32((*mem)[1]), N[X], N[Y], N[Z], pbc, cfg)
	} else {
		G := math.Exp(-kappa_temp*ctime) * (math.Cos(wc_temp*ctime)*(x0-msatCell*(*mem)[0]) - math.Sin(wc_temp*ctime)*(p0-msatCell*(*mem)[1]))

		Madd2(dst, dst, brms, 1.0, float32(G))
	}
}

// func AddCavity2(dst, m, brms *data.Slice, wc, kappa MSlice, x0, p0, msatCell, deltah, ctime float64, mem *[2]float64, mesh *data.Mesh) {
//
// 	N := mesh.Size()
//
// 	kappa_temp := float64(kappa.Mul(0))
// 	wc_temp := float64(wc.Mul(0))
//
// 	brms = data.Resample(brms, N) // reshape of OVF Brms file to mesh size
// 	brms_m := float64(Dot(brms, m))
//
// 	(*mem)[0] += math.Exp(kappa_temp*ctime) * math.Sin(wc_temp*ctime) * brms_m * deltah
// 	(*mem)[1] += math.Exp(kappa_temp*ctime) * math.Cos(wc_temp*ctime) * brms_m * deltah
//
// 	G := math.Exp(-kappa_temp*ctime) * (math.Cos(wc_temp*ctime)*(x0-msatCell*(*mem)[0]) - math.Sin(wc_temp*ctime)*(p0-msatCell*(*mem)[1]))
//
// 	Madd2(dst, dst, brms, 1.0, float32(G))
// }
