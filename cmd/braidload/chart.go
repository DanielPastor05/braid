package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// SVG charts, written by hand, because a README of an inference server with no
// curves in it is telling you the author never had to show performance to anyone
// who would not read the table.
//
// No dependency: go.mod has none and a chart is a path element. The colours come
// from a prefers-color-scheme block inside the document, so the same file reads
// on GitHub's light and dark themes -- an image with a white plot area is a
// white rectangle in the middle of a dark page.

type point struct{ x, y float64 }

type series struct {
	label  string
	colour string
	points []point
	// labels annotate individual points, e.g. the concurrency a throughput
	// and latency pair came from. Empty means none.
	labels []string
}

// palette is chosen for legibility on both themes rather than for prettiness:
// mid-saturation, and distinguishable without relying on hue alone for anyone
// who cannot separate red from green.
var palette = []string{"#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"}

type chart struct {
	title          string
	xLabel, yLabel string
	series         []series
	// xLog draws the x axis on a log scale, which is what a concurrency sweep
	// doubling at every level needs to not pile up against the left edge.
	xLog bool
	// xTicks and yTicks override the automatic ones when the natural values are
	// more informative than round numbers.
	xTicks []float64
}

const (
	chartWidth  = 720
	chartHeight = 420
	padLeft     = 70
	padRight    = 130 // the legend lives here
	padTop      = 46
	padBottom   = 56
)

func (c chart) render() string {
	var b strings.Builder

	plotW := float64(chartWidth - padLeft - padRight)
	plotH := float64(chartHeight - padTop - padBottom)

	minX, maxX, minY, maxY := c.bounds()
	if c.xLog {
		minX, maxX = math.Log10(math.Max(minX, 1e-9)), math.Log10(math.Max(maxX, 1e-9))
	}
	// A flat series would otherwise divide by zero and draw a line through the
	// axis label.
	if maxX == minX {
		maxX = minX + 1
	}
	if maxY == minY {
		maxY = minY + 1
	}
	// Headroom, so the topmost point is not welded to the frame.
	maxY += (maxY - minY) * 0.08

	toX := func(v float64) float64 {
		if c.xLog {
			v = math.Log10(math.Max(v, 1e-9))
		}
		return float64(padLeft) + (v-minX)/(maxX-minX)*plotW
	}
	toY := func(v float64) float64 {
		return float64(padTop) + plotH - (v-minY)/(maxY-minY)*plotH
	}

	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" `+
		`viewBox="0 0 %d %d" font-family="ui-sans-serif,system-ui,sans-serif">`,
		chartWidth, chartHeight, chartWidth, chartHeight)

	// The theme block. Everything that must flip is a class rather than an
	// attribute, because an attribute wins over a stylesheet.
	b.WriteString(`<style>
  .bg{fill:#ffffff}.ink{fill:#111827}.muted{fill:#6b7280}
  .grid{stroke:#e5e7eb}.axis{stroke:#9ca3af}
  @media (prefers-color-scheme: dark){
    .bg{fill:#0d1117}.ink{fill:#e6edf3}.muted{fill:#8b949e}
    .grid{stroke:#21262d}.axis{stroke:#484f58}
  }
  text{font-size:12px}.title{font-size:15px;font-weight:600}
</style>`)

	fmt.Fprintf(&b, `<rect class="bg" width="%d" height="%d"/>`, chartWidth, chartHeight)
	fmt.Fprintf(&b, `<text class="title ink" x="%d" y="24">%s</text>`, padLeft-40, escape(c.title))

	// Horizontal gridlines with their labels.
	for i := range 6 {
		v := minY + (maxY-minY)*float64(i)/5
		y := toY(v)
		fmt.Fprintf(&b, `<line class="grid" x1="%d" y1="%.1f" x2="%.1f" y2="%.1f"/>`,
			padLeft, y, float64(padLeft)+plotW, y)
		fmt.Fprintf(&b, `<text class="muted" x="%d" y="%.1f" text-anchor="end">%s</text>`,
			padLeft-8, y+4, tidy(v))
	}

	ticks := c.xTicks
	if ticks == nil {
		for i := range 5 {
			v := minX + (maxX-minX)*float64(i)/4
			if c.xLog {
				v = math.Pow(10, v)
			}
			ticks = append(ticks, v)
		}
	}
	for _, v := range ticks {
		x := toX(v)
		fmt.Fprintf(&b, `<text class="muted" x="%.1f" y="%.1f" text-anchor="middle">%s</text>`,
			x, float64(padTop)+plotH+18, tidy(v))
	}

	fmt.Fprintf(&b, `<line class="axis" x1="%d" y1="%.1f" x2="%.1f" y2="%.1f"/>`,
		padLeft, float64(padTop)+plotH, float64(padLeft)+plotW, float64(padTop)+plotH)
	fmt.Fprintf(&b, `<line class="axis" x1="%d" y1="%d" x2="%d" y2="%.1f"/>`,
		padLeft, padTop, padLeft, float64(padTop)+plotH)

	fmt.Fprintf(&b, `<text class="muted" x="%.1f" y="%d" text-anchor="middle">%s</text>`,
		float64(padLeft)+plotW/2, chartHeight-14, escape(c.xLabel))
	fmt.Fprintf(&b,
		`<text class="muted" x="%.1f" y="%.1f" text-anchor="middle" transform="rotate(-90 %.1f %.1f)">%s</text>`,
		18.0, float64(padTop)+plotH/2, 18.0, float64(padTop)+plotH/2, escape(c.yLabel))

	for i, s := range c.series {
		colour := s.colour
		if colour == "" {
			colour = palette[i%len(palette)]
		}

		var path strings.Builder
		for j, p := range s.points {
			verb := "L"
			if j == 0 {
				verb = "M"
			}
			fmt.Fprintf(&path, "%s%.1f %.1f ", verb, toX(p.x), toY(p.y))
		}
		fmt.Fprintf(&b, `<path d="%s" fill="none" stroke="%s" stroke-width="2"/>`,
			strings.TrimSpace(path.String()), colour)

		for j, p := range s.points {
			fmt.Fprintf(&b, `<circle cx="%.1f" cy="%.1f" r="3.5" fill="%s"/>`,
				toX(p.x), toY(p.y), colour)
			if j < len(s.labels) && s.labels[j] != "" {
				fmt.Fprintf(&b,
					`<text class="muted" x="%.1f" y="%.1f" text-anchor="middle">%s</text>`,
					toX(p.x), toY(p.y)-10, escape(s.labels[j]))
			}
		}

		y := float64(padTop + 6 + i*20)
		x := float64(chartWidth - padRight + 16)
		fmt.Fprintf(&b, `<rect x="%.1f" y="%.1f" width="10" height="10" fill="%s" rx="2"/>`,
			x, y, colour)
		fmt.Fprintf(&b, `<text class="ink" x="%.1f" y="%.1f">%s</text>`,
			x+16, y+9, escape(s.label))
	}

	b.WriteString(`</svg>`)
	return b.String()
}

func (c chart) bounds() (minX, maxX, minY, maxY float64) {
	minX, minY = math.Inf(1), math.Inf(1)
	maxX, maxY = math.Inf(-1), math.Inf(-1)
	for _, s := range c.series {
		for _, p := range s.points {
			minX, maxX = math.Min(minX, p.x), math.Max(maxX, p.x)
			minY, maxY = math.Min(minY, p.y), math.Max(maxY, p.y)
		}
	}
	if math.IsInf(minX, 1) {
		return 0, 1, 0, 1
	}
	// Y starts at zero unless the data is far from it: a throughput chart whose
	// axis starts at 300 exaggerates every difference on it, which is the oldest
	// way to mislead with a true graph.
	if minY > 0 && minY < maxY*0.6 {
		minY = 0
	}
	return minX, maxX, minY, maxY
}

func tidy(v float64) string {
	switch {
	case v == math.Trunc(v) && math.Abs(v) < 100000:
		return fmt.Sprintf("%.0f", v)
	case math.Abs(v) < 10:
		return fmt.Sprintf("%.2f", v)
	default:
		return fmt.Sprintf("%.1f", v)
	}
}

func escape(s string) string {
	return strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;").Replace(s)
}

// writeCharts turns a finished sweep into the three pictures the README wants.
//
// They are the three questions a reader of an inference server's numbers
// actually has: how much throughput costs how much tail latency, how much of
// that throughput is served fast enough to count, and what the distribution
// behind a percentile looks like.
func writeCharts(dir string, levels []level, sloMS float64) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	knee := series{label: "concurrency"}
	good := []point{}
	raw := []point{}
	for _, l := range levels {
		knee.points = append(knee.points, point{x: l.ttft99, y: l.rate})
		knee.labels = append(knee.labels, fmt.Sprintf("%d", l.clients))
		good = append(good, point{x: float64(l.clients), y: l.goodput})
		raw = append(raw, point{x: float64(l.clients), y: l.rate})
	}

	charts := map[string]chart{
		"throughput-vs-tail.svg": {
			title:  "What throughput costs in tail latency",
			xLabel: "time to first token, p99 (ms)",
			yLabel: "tokens/s",
			series: []series{knee},
			xLog:   true,
		},
		"goodput.svg": {
			title:  fmt.Sprintf("Throughput, and the part of it served within %.0f ms", sloMS),
			xLabel: "concurrent clients",
			yLabel: "tokens/s",
			series: []series{
				{label: "all tokens", points: raw},
				{label: fmt.Sprintf("under %.0f ms", sloMS), points: good},
			},
			xLog:   true,
			xTicks: clientTicks(levels),
		},
	}

	// Not every level: eight overlapping curves is a smear, and the point of a
	// CDF here is to show what a percentile summarises rather than to plot
	// everything measured. Four spread across the sweep -- the quiet end, the
	// knee, and past it -- is what a reader can actually separate.
	var cdf []series
	for _, l := range pickForCDF(levels) {
		if len(l.ttfts) == 0 {
			continue
		}
		cdf = append(cdf, series{
			label:  fmt.Sprintf("%d clients", l.clients),
			points: toCDF(l.ttfts),
		})
	}
	if len(cdf) > 0 {
		charts["latency-cdf.svg"] = chart{
			title:  "Where the percentiles come from",
			xLabel: "time to first token (ms)",
			yLabel: "share of requests (%)",
			series: cdf,
			xLog:   true,
		}
	}

	for name, c := range charts {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(c.render()), 0o644); err != nil {
			return fmt.Errorf("writing %s: %w", name, err)
		}
	}
	return nil
}

func clientTicks(levels []level) []float64 {
	out := make([]float64, 0, len(levels))
	for _, l := range levels {
		out = append(out, float64(l.clients))
	}
	return out
}

// toCDF turns a set of samples into an empirical distribution, thinned to at
// most a couple of hundred points so the file stays small and the line stays a
// line rather than a smear.
func toCDF(samples []time.Duration) []point {
	sorted := append([]time.Duration(nil), samples...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	const want = 160
	step := max(1, len(sorted)/want)

	out := make([]point, 0, want+1)
	for i := 0; i < len(sorted); i += step {
		out = append(out, point{
			x: float64(sorted[i].Microseconds()) / 1000,
			y: 100 * float64(i+1) / float64(len(sorted)),
		})
	}
	// The last sample is the one a p99 reader cares about, so it is never the
	// one thinning drops.
	last := sorted[len(sorted)-1]
	out = append(out, point{x: float64(last.Microseconds()) / 1000, y: 100})
	return out
}

// pickForCDF spreads at most four levels across the sweep, always keeping the
// first and the last: the quiet end and the overloaded one are the two a reader
// is comparing, and everything interesting is between them.
func pickForCDF(levels []level) []level {
	const want = 4
	if len(levels) <= want {
		return levels
	}
	out := make([]level, 0, want)
	for i := range want {
		out = append(out, levels[i*(len(levels)-1)/(want-1)])
	}
	return out
}
