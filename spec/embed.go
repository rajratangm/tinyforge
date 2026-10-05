// Package spec embeds the JobSpec JSON Schema so forgectl validates against the same file the docs describe.
package spec

import _ "embed"

//go:embed jobspec.v1alpha1.schema.json
var JobSpecSchema []byte
