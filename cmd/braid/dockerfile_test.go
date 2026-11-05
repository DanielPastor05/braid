package main

import (
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"
)

// The image is only built weekly, because its base is five gigabytes, and neither
// of the two things pinned here shows up in a build anyway: they fail when the
// container runs. So they are checked against the Dockerfile on every push.

func dockerfile(t *testing.T) []string {
	t.Helper()
	body, err := os.ReadFile("../../Dockerfile")
	if err != nil {
		t.Fatalf("reading the Dockerfile: %v", err)
	}
	return strings.Split(strings.ReplaceAll(string(body), "\r\n", "\n"), "\n")
}

// TestArgumentsAddToTheImagesFlags: `docker run image -auth-token x` replaces CMD
// whole. When the flags that select the worker lived in CMD, the README's own
// command dropped them, the server fell through to its mock backend, and it
// answered every request with invented text at excellent latencies.
//
// Later flags win in Go's flag package, so -addr can still be overridden.
func TestArgumentsAddToTheImagesFlags(t *testing.T) {
	var entrypoint []string
	for _, line := range dockerfile(t) {
		if strings.HasPrefix(line, "CMD ") {
			t.Errorf("a CMD is replaced by any argument to docker run: %s", line)
		}
		if rest, ok := strings.CutPrefix(line, "ENTRYPOINT "); ok {
			if err := json.Unmarshal([]byte(rest), &entrypoint); err != nil {
				t.Fatalf("ENTRYPOINT is not the JSON form: %v", err)
			}
		}
	}
	for _, flag := range []string{"-worker", "-model", "-addr"} {
		if !slices.Contains(entrypoint, flag) {
			t.Errorf("ENTRYPOINT %v does not set %s", entrypoint, flag)
		}
	}
}

// TestTheHealthCheckIsTheProbe: the check used to run the load harness with
// -requests 0, which exits 1 before it asks anything.
func TestTheHealthCheckIsTheProbe(t *testing.T) {
	lines := dockerfile(t)
	for i, line := range lines {
		if !strings.HasPrefix(line, "HEALTHCHECK ") {
			continue
		}
		if i+1 >= len(lines) || !strings.Contains(lines[i+1], `"-health"`) {
			t.Errorf("the check does not run the probe: %s", lines[i+1])
		}
		return
	}
	t.Error("the image has no HEALTHCHECK")
}
