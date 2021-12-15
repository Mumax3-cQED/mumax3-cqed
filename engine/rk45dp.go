package engine

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// var count int32

type RK45DP struct {
	k1 *data.Slice // torque at end of step is kept for beginning of next step
}

func (rk *RK45DP) Step() {
	m := M.Buffer()
	size := m.Size()

	// if cuda.M_rk45 == nil {
	// 	cuda.M_rk45 = make([][]float32, 0)
	// }

	cuda.M_rk = cuda.InitRKStepArray(cuda.M_rk, size)

	if FixDt != 0 {
		Dt_si = FixDt
	}

	// upon resize: remove wrongly sized k1
	if rk.k1.Size() != m.Size() {
		rk.Free()
	}

	// first step ever: one-time k1 init and eval
	if rk.k1 == nil {
		rk.k1 = cuda.NewSlice(3, size)
		// log.Println("buffer1: ", M.Buffer().Size())
		torqueFn(rk.k1)
	}

	// FSAL cannot be used with finite temperature
	if !Temp.isZero() {
		torqueFn(rk.k1)
	}

	t0 := Time
	// backup magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)
	// cuda.MdataTemp(cuda.M_rk, m, Time)

	k2, k3, k4, k5, k6 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)
	defer cuda.Recycle(k5)
	defer cuda.Recycle(k6)
	// k2 will be re-used as k7

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL

	// there is no explicit stage 1: k1 from previous step

	// stage 2
	Time = t0 + (1./5.)*Dt_si
	cuda.Madd2(m, m, rk.k1, 1, (1./5.)*h) // m = m*1 + k1*h/5

	// log.Println(m.Comp(0))
	// cuda.MdataTemp(cuda.M_rk, m, Time)
	// cuda.PrintRes(cuda.M_rk)
	// cuda.AppendData(m, Time, cuda.M_rk45)
	// log.Println("m stage2:", cuda.GetElem(m, 0, 0))
	// log.Println("Time stage2:", Time)
	M.normalize()
	torqueFn(k2)

	// stage 3
	Time = t0 + (3./10.)*Dt_si
	cuda.Madd3(m, m0, rk.k1, k2, 1, (3./40.)*h, (9./40.)*h)
	// cuda.MdataTemp(cuda.M_rk, m, Time)
	// cuda.AppendData(m, Time, cuda.M_rk)
	// log.Println("m stage3:", cuda.GetElem(m, 0, 0))
	// log.Println("Time stage3:", Time)
	M.normalize()
	torqueFn(k3)

	// stage 4
	Time = t0 + (4./5.)*Dt_si
	madd4(m, m0, rk.k1, k2, k3, 1, (44./45.)*h, (-56./15.)*h, (32./9.)*h)
	// cuda.MdataTemp(cuda.M_rk, m, Time)
	// cuda.AppendData(m, Time, cuda.M_rk)
	// log.Println("m stage4:", cuda.GetElem(m, 0, 0))
	// log.Println("Time stage4:", Time)
	M.normalize()
	torqueFn(k4)

	// stage 5
	Time = t0 + (8./9.)*Dt_si
	madd5(m, m0, rk.k1, k2, k3, k4, 1, (19372./6561.)*h, (-25360./2187.)*h, (64448./6561.)*h, (-212./729.)*h)
	// cuda.MdataTemp(cuda.M_rk, m, Time)
	// cuda.AppendData(m, Time, cuda.M_rk)
	// log.Println("m stage5:", cuda.GetElem(m, 0, 0))
	// log.Println("Time stage5:", Time)
	M.normalize()
	torqueFn(k5)

	// stage 6
	Time = t0 + (1.)*Dt_si
	madd6(m, m0, rk.k1, k2, k3, k4, k5, 1, (9017./3168.)*h, (-355./33.)*h, (46732./5247.)*h, (49./176.)*h, (-5103./18656.)*h)
	// cuda.MdataTemp(cuda.M_rk, m, Time)
	// cuda.AppendData(m, Time, cuda.M_rk)
	// log.Println("m stage6:", cuda.GetElem(m, 0, 0))
	// log.Println("Time stage6:", Time)
	M.normalize()
	torqueFn(k6)

	// stage 7: 5th order solution
	Time = t0 + (1.)*Dt_si
	// no k2
	madd6(m, m0, rk.k1, k3, k4, k5, k6, 1, (35./384.)*h, (500./1113.)*h, (125./192.)*h, (-2187./6784.)*h, (11./84.)*h) // 5th
	cuda.MdataTemp(cuda.M_rk, m, Time)
	// cuda.AppendData(m, Time, cuda.M_rk)
	M.normalize()
	k7 := k2     // re-use k2
	torqueFn(k7) // next torque if OK

	// error estimate
	Err := cuda.Buffer(3, size) //k3 // re-use k3 as error estimate
	defer cuda.Recycle(Err)
	madd6(Err, rk.k1, k3, k4, k5, k6, k7, (35./384.)-(5179./57600.), (500./1113.)-(7571./16695.), (125./192.)-(393./640.), (-2187./6784.)-(-92097./339200.), (11./84.)-(187./2100.), (0.)-(1./40.))

	// determine error
	err := cuda.MaxVecNorm(Err) * float64(h)
	// count += 1
	// log.Println("pasa: ", count)
	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		setLastErr(err)
		setMaxTorque(k7)
		NSteps++
		Time = t0 + Dt_si
		adaptDt(math.Pow(MaxErr/err, 1./5.))
		data.Copy(rk.k1, k7) // FSAL
	} else {
		// undo bad step
		//util.Println("Bad step at t=", t0, ", err=", err)
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./6.))
	}
}

func (rk *RK45DP) Free() {
	rk.k1.Free()
	rk.k1 = nil
}

// TODO: into cuda
func madd5(dst, src1, src2, src3, src4, src5 *data.Slice, w1, w2, w3, w4, w5 float32) {
	cuda.Madd3(dst, src1, src2, src3, w1, w2, w3)
	cuda.Madd3(dst, dst, src4, src5, 1, w4, w5)
}

func madd6(dst, src1, src2, src3, src4, src5, src6 *data.Slice, w1, w2, w3, w4, w5, w6 float32) {
	madd5(dst, src1, src2, src3, src4, src5, w1, w2, w3, w4, w5)
	cuda.Madd2(dst, dst, src6, 1, w6)
}
