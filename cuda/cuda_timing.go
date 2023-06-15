package cuda

// CREATED AND MODIFIED INMA
import (
	"github.com/mumax/3/data"
)

func GetElemPos(slice *data.Slice, position int) float32 {
	mz_temp := GetCell(slice, position, 0, 0, 0)

	return mz_temp
}
