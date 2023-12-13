package engine

// CREATED AND MODIFIED INMA
import (
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

func leapYears(date time.Time) (leaps int) {

	y, m, _ := date.Date()

	if m <= 2 {
		y--
	}

	leaps = y/4 + y/400 - y/100

	return leaps
}

func getTimeDifference(a time.Time) (days, hours, minutes, seconds int) {

	b := time.Now()

	monthDays := [12]int{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}

	y1, m1, d1 := a.Date()
	y2, m2, d2 := b.Date()
	h1, min1, s1 := a.Clock()
	h2, min2, s2 := b.Clock()

	totalDays1 := y1*365 + d1

	for i := 0; i < (int)(m1)-1; i++ {
		totalDays1 += monthDays[i]
	}

	totalDays1 += leapYears(a)
	totalDays2 := y2*365 + d2

	for i := 0; i < (int)(m2)-1; i++ {
		totalDays2 += monthDays[i]
	}

	totalDays2 += leapYears(b)

	days = totalDays2 - totalDays1
	hours = h2 - h1
	minutes = min2 - min1
	seconds = s2 - s1

	if seconds < 0 {
		seconds += 60
		minutes--
	}

	if minutes < 0 {
		minutes += 60
		hours--
	}

	if hours < 0 {
		hours += 24
		days--
	}

	return days, hours, minutes, seconds
}
