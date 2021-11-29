package cuda

var (
  Time_cuda float32
  Dt_cuda float32
  Wc_cuda float32
  Brms_cuda [3]float64
)

func SetTimingCuda(time float64, dt float64) {
  Time_cuda = float32(time)
  Dt_cuda = float32(dt)
}

func SetBrms(brms [3]float64) {
  Brms_cuda = brms
}

func SetWc(wc float64){
  Wc_cuda = float32(wc)
}
