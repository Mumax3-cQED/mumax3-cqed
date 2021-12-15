package cuda

import (
	"github.com/mumax/3/data"
)

var (
	Time_cuda     float32
	Fixed_dt_cuda float32
	Wc_cuda       float32
	Brms_cuda     [3]float32
	// Stop_time_cuda float32
	// Step_Times *data.Slice
	// M_rk45     [][]float32
	M_rk *data.Slice
)

// func SetStepTimes(torqueDst *data.Slice) {
// 	Step_Times = torqueDst
// }

func SetDtCuda(dt float64) {

	Fixed_dt_cuda = float32(dt)
}

func SetTimeCuda(time float64) {

	Time_cuda = float32(time)
}

// func SetStopTime(stop_time float64) {
// 	Stop_time_cuda = float32(stop_time)
// }

func SetBrms(brms [3]float64) {
	Brms_cuda = [3]float32{float32(brms[0]), float32(brms[1]), float32(brms[2])}
}

func SetWc(wc float64) {
	Wc_cuda = float32(wc)
}

// func PrintRes(mm *data.Slice) {
// 	// comp := mm.Comp(0)
// 	// vv := GetCell(mm, 0, 0, 0, 0)
//
// 	// mm.Vectors()
// 	// log.Println(mm.Size())
// }

func InitRKStepArray(rk_step *data.Slice, size [3]int) *data.Slice {

	if rk_step == nil {
		rk_step = NewSlice(4, size)
	}

	return rk_step
}

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
