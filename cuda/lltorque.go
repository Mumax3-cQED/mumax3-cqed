package cuda

import (
	"github.com/mumax/3/data"
)

// Landau-Lifshitz torque divided by gamma0:
// 	- 1/(1+α²) [ m x B +  α m x (m x B) ]
// 	torque in Tesla
// 	m normalized
// 	B in Tesla
// see lltorque.cu
func LLTorque(torque, m, B *data.Slice, alpha MSlice) {
	N := torque.Len()
	cfg := make1DConf(N)
	//Ext := 1
	//		log.Println("Exec_threads before: ", Exec_threads)
	// k_lltorque2_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
	// 	m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
	// 	B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
	// 	alpha.DevPtr(0), alpha.Mul(0), N, util.Bextra_vector, cfg)
	// log.Println("medidas: ", size[0], " ", size[1], " ", size[2])

	// 	var h_bar float64 = 1.054571817e-34 // h-bar planck value
	// 	var muB float64 = 9.274009994e-24 // Bohr magneton
	// 	var gs float64 = 2.0
	//
	// 	var constant_term float64 = math.Pow(gs,2)*math.Pow(muB,2) //(float)(powf(gs,2)*powf(muB,2))/powf(h_bar,3);
	// 	constant_term = constant_term / math.Pow(h_bar, 3)
	//
	// log.Println("constant_term: ", float64(constant_term))
	//log.Println("venga ahi!!!")
	//log.Println("Stop time: ", Stop_time_cuda)
	//log.Println("Time_cuda: ", Time_cuda)

	k_lltorque2_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		alpha.DevPtr(0), alpha.Mul(0), N, Time_cuda, Fixed_dt_cuda, Stop_time_cuda, Wc_cuda, Brms_cuda[0], Brms_cuda[1], Brms_cuda[2], Step_Times.DevPtr(X), cfg)

	//		Exec_threads += 1;
	// log.Println("Exec_threads AFTERRRR: ", Exec_threads)
	//			var v float64 =  *((*float64)(m.DevPtr(Y)))
	//log.Println(" m.DevPtr(Y):", v)
	// value := *((*int32)( unsafe.Pointer(&Si_sum_total)))
	//	  log.Println("lltorque2_args.arg_si_sum_total: ", value)
	// debug.PrintStack()
	//debug.PrintStack()
}

// Landau-Lifshitz torque with precession disabled.
// Used by engine.Relax().
func LLNoPrecess(torque, m, B *data.Slice) {
	N := torque.Len()
	cfg := make1DConf(N)

	k_llnoprecess_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z), N, cfg)
}
