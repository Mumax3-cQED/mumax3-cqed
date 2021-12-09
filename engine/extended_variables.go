package engine

import "github.com/mumax/3/util"

func init() {

	DeclVar("B_rms", &util.Brms_vector, "Brms Extra parameter")
	DeclVar("Wc", &util.Wc, "Wc Extra parameter")
}
