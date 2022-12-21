package cuda

// MODIFIED INMA
import (
	"github.com/mumax/3/data"
)

func GetZElem(slice *data.Slice) float32 {
	mz_temp := GetCell(slice, 2, 0, 0, 0)

	return mz_temp
}

func GetElemPos(slice *data.Slice, position int) float32 {
	mz_temp := GetCell(slice, position, 0, 0, 0)

	return mz_temp
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
