package cuda

import "github.com/mumax/3/data"

var (
	Time_cuda      *data.Slice
	Fixed_dt_cuda  *data.Slice
	Wc_cuda        float32
	Brms_cuda      [3]float32
	Stop_time_cuda float32
	Step_Times     *data.Slice
)

func SetStepTimes(torqueDst *data.Slice) {
	Step_Times = torqueDst
}

func SetDtCuda(dt float64) {

	SetElem(Fixed_dt_cuda, 0, 0, float32(dt))
}

func SetTimeCuda(time float64) {

	SetElem(Time_cuda, 0, 0, float32(time))
}

func SetStopTime(stop_time float64) {
	Stop_time_cuda = float32(stop_time)
}

func SetBrms(brms [3]float64) {
	Brms_cuda = [3]float32{float32(brms[0]), float32(brms[1]), float32(brms[2])}
}

func SetWc(wc float64) {
	Wc_cuda = float32(wc)
}
