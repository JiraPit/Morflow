package morflow

import "errors"

// Error is an error reported by the Morflow engine: a pipeline that does not
// parse or compile, an action that is missing, or a run that fails. Message
// is the engine's own message, the same in every Morflow package.
type Error struct {
	Message string
}

func (e *Error) Error() string { return e.Message }

// ErrClosed is returned by runs on a pipeline after Close.
var ErrClosed = errors.New("morflow: the pipeline has been closed")
