package cuda

// MODIFIED INMA
import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// multiply: dst[i] = a[i] * b[i]
// a and b must have the same number of components
func Mul(dst, a, b *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_mul_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), N, cfg)
	}
}

// divide: dst[i] = a[i] / b[i]
// divide-by-zero yields zero.
func Div(dst, a, b *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_pointwise_div_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), N, cfg)
	}
}

// Add: dst = src1 + src2.
func Add(dst, src1, src2 *data.Slice) {
	Madd2(dst, src1, src2, 1, 1)
}

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2
func Madd2(dst, src1, src2 *data.Slice, factor1, factor2 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src2.Len() == N)
	util.Assert(src1.NComp() == nComp && src2.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_madd2_async(dst.DevPtr(c), src1.DevPtr(c), factor1,
			src2.DevPtr(c), factor2, N, cfg)
	}
}

func CalcMSpinTorque(dst, m_current *data.Slice, ctime float64, deltah float32, brms []float64, wc float64) {

	N := dst.Len()
	util.Assert(m_current.Len() == N)

	nComp := dst.NComp()
	util.Assert(nComp >= m_current.NComp())
	cfg := make1DConf(N)

	size := m_current.Size()
	Nx := size[0]
	Ny := size[1]
	Nz := size[2]

	k_mdatatemp_async(dst.DevPtr(0), dst.DevPtr(1), dst.DevPtr(2), dst.DevPtr(3), dst.DevPtr(4), dst.DevPtr(5), dst.DevPtr(6), dst.DevPtr(7), dst.DevPtr(8), dst.DevPtr(9), dst.DevPtr(10), dst.DevPtr(11),
		m_current.DevPtr(0), m_current.DevPtr(1), m_current.DevPtr(2),
		float32(ctime), deltah, float32(brms[0]), float32(brms[1]), float32(brms[2]), float32(wc), Nx, Ny, Nz, cfg)
}

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2 + src3 * factor3
func Madd3(dst, src1, src2, src3 *data.Slice, factor1, factor2, factor3 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src2.Len() == N && src3.Len() == N)
	util.Assert(src1.NComp() == nComp && src2.NComp() == nComp && src3.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_madd3_async(dst.DevPtr(c), src1.DevPtr(c), factor1,
			src2.DevPtr(c), factor2, src3.DevPtr(c), factor3, N, cfg)
	}
}
