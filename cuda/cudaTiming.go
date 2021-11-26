package cuda

var (
	Time_cuda float32
	Dt_cuda   float32
)

func SetTimingCuda(time float64, dt float64) {
	Time_cuda = float32(time)
	Dt_cuda = float32(dt)
}
