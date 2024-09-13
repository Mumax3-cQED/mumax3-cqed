package cuda

// Add cavity field to Beff
import (
	"math"

	"github.com/mumax/3/data"
)

// Add exchange field to Beff.
// 	full_m: normalized magnetization
//  brms: Zero point magnetic field in Tesla
//  wc: Cavity frequency in rad/s
// 	kappa: Cavity dissipation in rad/s
//  x0: Cavity initial condtion
//  p0: Cavity initial condtion
// see cavity.cu
func AddCavity(dst, full_m, brms *data.Slice, wc, kappa, x0, p0, vc2_hbar, dt, ctime float64, mem *[2]float32, mesh *data.Mesh) {

	N := mesh.Size()
	brms = data.Resample(brms, N) // reshape of OVF Brms file to mesh size
	brms_m := Dot(brms, full_m)

	(*mem)[0] += float32(math.Exp(kappa*ctime)*math.Sin(wc*ctime)) * brms_m * float32(dt)
	(*mem)[1] += float32(math.Exp(kappa*ctime)*math.Cos(wc*ctime)) * brms_m * float32(dt)

	G := float32(math.Exp(-kappa*ctime)) * (float32(math.Cos(wc*ctime))*(float32(x0-vc2_hbar)*(*mem)[0]) - float32(math.Sin(wc*ctime))*(float32(p0-vc2_hbar)*(*mem)[1]))

	Madd2(dst, dst, brms, 1.0, G)
}
