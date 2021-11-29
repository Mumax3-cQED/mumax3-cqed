package engine

import "github.com/mumax/3/util"

func init() {

	//DeclVar("Ext_param", &util.Ext_param, "Extra parameter")
	//DeclVar("Bextra_vector", &util.Bextra_vector, "B Extra parameter")
  DeclVar("B_rms", &util.Brms_vector , "Brms Extra parameter")
	DeclVar("Wc", &util.Wc , "Wc Extra parameter")
}
