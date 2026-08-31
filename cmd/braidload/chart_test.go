package main

import (
	"encoding/xml"
	"math"
	"strings"
	"testing"
	"time"
)

// The charts go in the README, where a broken one is a broken image and nothing
// else -- no error, no stack, just a missing picture that everybody assumes is
// GitHub's fault. So the renderer gets checked like anything else that produces
// output somebody else parses.

func wellFormed(t *testing.T, svg string) {
	t.Helper()
	decoder := xml.NewDecoder(strings.NewReader(svg))
	for {
		_, err := decoder.Token()
		if err != nil {
			if err.Error() == "EOF" {
				return
			}
			t.Fatalf("the SVG is not well-formed XML: %v", err)
		}
	}
}

func TestAChartRendersWellFormedSVG(t *testing.T) {
	c := chart{
		title:  "Throughput & latency <test>",
		xLabel: "clients",
		yLabel: "tokens/s",
		series: []series{
			{label: "all", points: []point{{1, 340}, {16, 1900}, {64, 3003}}},
			{label: "in time", points: []point{{1, 340}, {16, 1800}, {64, 1200}}},
		},
		xLog: true,
	}

	svg := c.render()
	wellFormed(t, svg)

	if !strings.HasPrefix(svg, "<svg ") || !strings.HasSuffix(svg, "</svg>") {
		t.Error("the output is not a single svg element")
	}
	// The title carries characters that would break the document if they were
	// not escaped, which is the mistake string concatenation makes.
	if strings.Contains(svg, "<test>") {
		t.Error("the title was interpolated without escaping")
	}
	if !strings.Contains(svg, "&lt;test&gt;") {
		t.Error("the title's angle brackets were not escaped")
	}
	// Both themes, or the chart is a white rectangle on a dark page.
	if !strings.Contains(svg, "prefers-color-scheme: dark") {
		t.Error("the chart has no dark-theme rules")
	}
	for _, label := range []string{"all", "in time", "clients", "tokens/s"} {
		if !strings.Contains(svg, label) {
			t.Errorf("%q does not appear in the output", label)
		}
	}
}

// TestAFlatSeriesDoesNotDivideByZero: a level where every point has the same
// value is not exotic -- an idle server produces one -- and the naive scaling
// puts a NaN in every coordinate, which renders as nothing at all.
func TestAFlatSeriesDoesNotDivideByZero(t *testing.T) {
	for _, c := range []chart{
		{title: "flat y", series: []series{{label: "s", points: []point{{1, 5}, {2, 5}, {3, 5}}}}},
		{title: "one point", series: []series{{label: "s", points: []point{{1, 5}}}}},
		{title: "flat x", series: []series{{label: "s", points: []point{{1, 5}, {1, 9}}}}},
		{title: "nothing", series: []series{{label: "s"}}},
	} {
		svg := c.render()
		wellFormed(t, svg)
		if strings.Contains(svg, "NaN") {
			t.Errorf("%s: the output contains NaN coordinates", c.title)
		}
		if strings.Contains(svg, "Inf") {
			t.Errorf("%s: the output contains infinite coordinates", c.title)
		}
	}
}

// TestTheYAxisStartsAtZeroWhenItShould is about honesty rather than rendering.
//
// An axis that starts just under the smallest value makes a 5% difference look
// like a doubling, which is the oldest way to mislead with a graph that contains
// only true numbers.
func TestTheYAxisStartsAtZeroWhenItShould(t *testing.T) {
	near := chart{series: []series{{points: []point{{1, 340}, {2, 3003}}}}}
	_, _, minY, _ := near.bounds()
	if minY != 0 {
		t.Errorf("a series spanning 340..3003 got a y axis starting at %g, not 0", minY)
	}

	// The exception: data genuinely far from zero, where starting at zero would
	// flatten everything into one line.
	far := chart{series: []series{{points: []point{{1, 1000}, {2, 1010}}}}}
	_, _, minY, _ = far.bounds()
	if minY == 0 {
		t.Error("a series spanning 1000..1010 was flattened against a zero axis")
	}
}

func TestTheCDFCoversTheWholeDistribution(t *testing.T) {
	samples := make([]time.Duration, 1000)
	for i := range samples {
		samples[i] = time.Duration(i+1) * time.Millisecond
	}

	cdf := toCDF(samples)
	if len(cdf) < 2 {
		t.Fatalf("the CDF has %d points", len(cdf))
	}
	if cdf[len(cdf)-1].y != 100 {
		t.Errorf("the CDF ends at %g%%, not 100", cdf[len(cdf)-1].y)
	}
	// The last sample is what a p99 reader came for, so thinning must never be
	// what drops it.
	if want := 1000.0; math.Abs(cdf[len(cdf)-1].x-want) > 0.001 {
		t.Errorf("the CDF's last x is %g ms, not the slowest sample at %g", cdf[len(cdf)-1].x, want)
	}
	for i := 1; i < len(cdf); i++ {
		if cdf[i].y < cdf[i-1].y {
			t.Fatalf("the CDF goes backwards at point %d", i)
		}
		if cdf[i].x < cdf[i-1].x {
			t.Fatalf("the CDF's x goes backwards at point %d: the samples were not sorted", i)
		}
	}
}
