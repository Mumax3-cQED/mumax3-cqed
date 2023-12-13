package engine

// CREATED AND MODIFIED INMA
import (
	"fmt"
	"strings"
	"time"
)

func getCurrentDate() (int, int, int, int, int, int) {

	date_current := time.Now()
	year, month, day := date_current.Date()
	hour := date_current.Hour()
	minute := date_current.Minute()
	seconds := date_current.Second()

	return year, int(month), day, hour, minute, seconds
}

func getTimeDifference(start time.Time) string {

	end := time.Now()

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
