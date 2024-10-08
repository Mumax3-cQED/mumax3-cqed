package engine

import (
	"fmt"
	"strings"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	ShowSimulationSummary           = true
	StartCheckpoint       time.Time = time.Now() // Starting date for mumax3 script to measure elapsed execution time, to set starting date anywhere in the script use --> StartCheckpoint = now()
)

func init() {
	DeclVar("StartCheckpoint", &StartCheckpoint, "Starting date for script launch (default now() at the beginning of mumax3 allocation)")
	DeclVar("ShowSimulationSummary", &ShowSimulationSummary, "Show simulation data summary after run() function (default=true)")
	DeclFunc("PrintScriptExecutionTime", PrintScriptExecutionTime, "Print and save to log the script execution time")
}

func If_Ternary(statement bool, a, b interface{}) interface{} {
	if statement {
		return a
	}
	return b
}

// Display script configuration summary in script output
// and insert this information into the log.txt file (see run.go)
func PrintParametersTimeEvolution(simulationTime *float64) {

	if IsCavityActive() {

		if !ShowSimulationSummary {
			return
		}

		util.AssertMsg(!Msat.isZero(), "saturation magnetization should not be 0")

		c, rec := B_rms.Slice()
		if rec {
			defer cuda.Recycle(c)
		}

		be, rec := B_ext.Slice()
		if rec {
			defer cuda.Recycle(be)
		}

		m_sat := Msat.MSlice()
		defer m_sat.Recycle()

		alpha := Alpha.MSlice()
		defer alpha.Recycle()

		LogIn("")
		LogIn("------------------------------------------------")

		year, month, day, hour, minute, seconds := getCurrentDate()
		full_date := fmt.Sprintf("%d-%02d-%02d %02d:%02d:%02d", year, month, day, hour, minute, seconds)

		LogIn(" Simulation date (yyyy-MM-dd HH:mm:ss):", full_date)
		LogIn(" Time evolution factor in LLG equation: Enabled")
		LogIn(" Cavity feature: Enabled")
		LogIn(" B_demag (magnetostatic field):", If_Ternary(EnableDemag, "Enabled", "Disabled").(string))
		LogIn(" Zhang-Li Spin-Transfer Torque:", If_Ternary(DisableZhangLiTorque, "Disabled", "Enabled").(string))
		LogIn(" Slonczewski Spin-Transfer Torque:", If_Ternary(DisableSlonczewskiTorque, "Disabled", "Enabled").(string))

		full_sizex, full_sizey, full_sizez, cell_size, num_cells := calcFullSize()

		LogIn(" Shape size (m):", full_sizex, "x", full_sizey, "x", full_sizez)
		LogIn(" Num. cells:", num_cells[X], "x", num_cells[Y], "x", num_cells[Z])
		LogIn(" Cell size (m):", cell_size[X], "x", cell_size[Y], "x", cell_size[Z])

		LogIn(" Kappa (rad/s):", Kappa)

		if alpha.Mul(0) != 0 {
			LogIn(" Alpha:", alpha.Mul(0))
		}

		LogIn(" Msat (A/m):", If_Ternary(m_sat.Mul(0) != 0, m_sat.Mul(0), 0.0).(float32))
		LogIn(" Cavity initial condition X0:", X0)
		LogIn(" Cavity initial condition P0:", P0)
		LogIn(" GammaLL (rad/Ts):", GammaLL)
		LogIn(" Wc (rad/s):", Wc)

		if uniform_vector.X() != 0.0 || uniform_vector.Y() != 0.0 || uniform_vector.Z() != 0.0 {
			LogIn(" Uniform vector (T): [", uniform_vector.X(), ",", uniform_vector.Y(), ",", uniform_vector.Z(), "]")
		}

		if c != nil {
			LogIn(" B_rms vector (T): [", getElemPos(c, X), ",", getElemPos(c, Y), ",", getElemPos(c, Z), "]")
		}

		if be != nil {
			LogIn(" B_ext vector (T): [", getElemPos(be, X), ",", getElemPos(be, Y), ",", getElemPos(be, Z), "]")
		}

		if FixDt != 0 {
			LogIn(" FixDt (s):", FixDt)
		}

		savePeriod := Table.autosave.period

		if savePeriod != 0 {
			LogIn(" Table autosave interval (s):", savePeriod)
		}

		if *simulationTime != 0 {
			LogIn(" Full simulation time (s):", *simulationTime)
		}

		LogIn("------------------------------------------------")
		LogIn("")
	}
}

// Get value at position from CUDA object
func getElemPos(slice *data.Slice, position int) float32 {
	mz_temp := cuda.GetCell(slice, position, 0, 0, 0)
	return mz_temp
}

// Obtain full size of shape
func calcFullSize() (float64, float64, float64, [3]float64, [3]int) {

	cell_size := Mesh().CellSize()
	size_cellx := cell_size[X]
	size_celly := cell_size[Y]
	size_cellz := cell_size[Z]

	num_cells := Mesh().Size()
	num_cellx := float64(num_cells[X])
	num_celly := float64(num_cells[Y])
	num_cellz := float64(num_cells[Z])

	full_sizex := size_cellx * num_cellx
	full_sizey := size_celly * num_celly
	full_sizez := size_cellz * num_cellz

	return full_sizex, full_sizey, full_sizez, cell_size, num_cells
}

// Get current date
func getCurrentDate() (int, int, int, int, int, int) {

	return getInfoFromTime(time.Now())
}

func getInfoFromTime(date_item time.Time) (int, int, int, int, int, int) {

	year, month, day := date_item.Date()
	hour := date_item.Hour()
	minute := date_item.Minute()
	seconds := date_item.Second()

	return year, int(month), day, hour, minute, seconds
}

// Print elapsed time of script execution, this function can be invoked at the end of mumax3 script
func PrintScriptExecutionTime() {

	diff_str := getTimeDifference(StartCheckpoint)

	LogIn("\n ---> Full mumax3 script elapsed time:", diff_str)

	year, month, day, hour, minute, seconds := getInfoFromTime(StartCheckpoint)
	full_date_start := fmt.Sprintf("%d-%02d-%02d %02d:%02d:%02d", year, month, day, hour, minute, seconds)

	LogIn(" ---> Start simulation date (yyyy-MM-dd HH:mm:ss):", full_date_start)

	year, month, day, hour, minute, seconds = getCurrentDate()
	full_date_end := fmt.Sprintf("%d-%02d-%02d %02d:%02d:%02d", year, month, day, hour, minute, seconds)

	LogIn(" ---> End simulation date (yyyy-MM-dd HH:mm:ss):", full_date_end, "\n")
}

func getTimeDifference(start time.Time) string {

	return getTimeDifferenceDates(start, time.Now())
}

// Get time difference between two dates with a given starting date
func getTimeDifferenceDates(start time.Time, end time.Time) string {

	if start.Location() != end.Location() {
		end = end.In(start.Location())
	}

	if start.After(end) {
		start, end = end, start
	}

	y1, M1, d1 := start.Date()
	y2, M2, d2 := end.Date()

	h1, m1, s1 := start.Clock()
	h2, m2, s2 := end.Clock()

	year := int(y2 - y1)
	month := int(M2 - M1)
	day := int(d2 - d1)
	hour := int(h2 - h1)
	min := int(m2 - m1)
	sec := int(s2 - s1)

	// Normalize negative values
	if sec < 0 {
		sec += 60
		min--
	}

	if min < 0 {
		min += 60
		hour--
	}

	if hour < 0 {
		hour += 24
		day--
	}

	if day < 0 {
		// days in month:
		t := time.Date(y1, M1, 32, 0, 0, 0, 0, time.UTC)
		day += 32 - t.Day()
		month--
	}

	if month < 0 {
		month += 12
		year--
	}

	return parseTimeResponse(year, month, day, hour, min, sec)
}

// Parse result function for elapsed time between dates
func parseTimeResponse(years, months, days, hours, mins, secs int) string {

	full_diff := ""

	if years > 0 {
		full_diff += fmt.Sprintf("%dy;", years)
	}

	if months > 0 {
		full_diff += fmt.Sprintf("%dM;", months)
	}

	if days > 0 {
		full_diff += fmt.Sprintf("%dd;", days)
	}

	if hours > 0 {
		full_diff += fmt.Sprintf("%dh;", hours)
	}

	if mins > 0 {
		full_diff += fmt.Sprintf("%dm;", mins)
	}

	if secs > 0 {
		full_diff += fmt.Sprintf("%ds", secs)
	}

	full_diff = strings.Replace(full_diff, ";", " ", -1)

	return full_diff
}
