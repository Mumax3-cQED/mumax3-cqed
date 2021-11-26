// package util provides common utilities for all other packages.
package util

import (
	"net"
	"path"
	"strings"
)

//var Ext_param int = 0
var Bextra_vector [3]float64

// Remove extension from file name.
func NoExt(file string) string {
	ext := path.Ext(file)
	return file[:len(file)-len(ext)]
}

// returns all network interface addresses, without CIDR mask
func InterfaceAddrs() []string {
	addrs, _ := net.InterfaceAddrs()
	ips := make([]string, 0, len(addrs))
	for _, addr := range addrs {
		IpCidr := strings.Split(addr.String(), "/")
		ips = append(ips, IpCidr[0])
	}
	return ips
}
