package cuda

// MODIFIED INMA
import (
	"github.com/mumax/3/data"
)

var (
	Fixed_dt_cuda float32 = 0.0
	Wc_cuda       float64
	Brms_cuda     []float64
	M_rk          *data.Slice = nil
	timeEvolution bool        = false
	CurrentTime   float64     = 0.0
	// brms_i        *data.Slice
	// sum_cells     *data.Slice
)

func SetCurrentTime(ctime float64) {
	CurrentTime = ctime
}

func SetDtCuda(dt float32) {
	Fixed_dt_cuda = dt
}

func SetTimeEvoStatus(enableTimeEvo bool) {
	timeEvolution = enableTimeEvo
}

// func IsBrmsZero(brms []float64) bool {
//
// 	util.Assert(len(brms) == 3)
// 	return (brms[0] == 0.0 && brms[1] == 0.0 && brms[2] == 0.0)
// }

func SetBrms(brms [3]float64) {

	Brms_cuda = make([]float64, 3)

	Brms_cuda[0] = brms[0]
	Brms_cuda[1] = brms[1]
	Brms_cuda[2] = brms[2]
}

func SetWc(wc float64) {

	Wc_cuda = wc
}

// func initBrmsSlice(size [3]int) *data.Slice {
//
// 	if brms_i == nil {
// 		brms_i = NewSlice(3, size)
// 	}
//
// 	return brms_i
// }
//
// func initSumSlice(size [3]int) *data.Slice {
//
// 	if sum_cells == nil {
// 		sum_cells = NewSlice(3, size)
// 	}
//
// 	return sum_cells
// }

func InitRKStepArray(size [3]int) *data.Slice {

	if M_rk == nil {
		M_rk = NewSlice(10, size)
	}

	return M_rk
}

// func GetZElem(slice *data.Slice) float32 {
// 	mz_temp := GetCell(slice, 2, 0, 0, 0)
//
// 	return mz_temp
// }

// func AppendData(m *data.Slice, time float64, destArray [][]float32) {
//
// 	size := m.Size()
//
// 	len_vec1 := size[0]
// 	len_vec2 := size[1]
// 	len_vec3 := size[2]
//
// 	var cell_idx int32 = 0
//
// 	for z := 0; z < len_vec3; z++ {
// 		for y := 0; y < len_vec2; y++ {
// 			for x := 0; x < len_vec1; x++ {
//
// 				mx_temp := GetCell(m, 0, x, y, z)
// 				my_temp := GetCell(m, 1, x, y, z)
// 				mz_temp := GetCell(m, 2, x, y, z)
//
// 				tmp := make([]float32, 0)
//
// 				tmp = append(tmp, float32(cell_idx))
// 				tmp = append(tmp, mx_temp)
// 				tmp = append(tmp, my_temp)
// 				tmp = append(tmp, mz_temp)
// 				tmp = append(tmp, float32(time))
// 				// tmp = append(tmp, float32(x))
// 				// tmp = append(tmp, float32(y))
// 				// tmp = append(tmp, float32(z))
//
// 				// log.Println("vvvv:", mx_temp, ", ", my_temp, ", ", mz_temp)
//
// 				destArray = append(destArray, tmp)
//
// 				cell_idx += 1
// 			}
// 		}
//
// 		// log.Println("vvvv:", len(destArray))
// 	}
//
// 	// log.Println("m stage7:", vec[2][0][0][1])
// 	// log.Println("comps:", comps)
// 	// log.Println("vec-1:", len(vec))
// 	// log.Println("vec0:", len(vec[0]))
// 	// log.Println("vec1:", len(vec[1]))
// 	// log.Println("vec2:", len(vec[1]))
// }
