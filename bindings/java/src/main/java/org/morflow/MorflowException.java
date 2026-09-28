package org.morflow;

/**
 * Exception thrown when a Morflow pipeline fails to parse, compile, or execute.
 */
public class MorflowException extends RuntimeException {
    public MorflowException(String message) {
        super(message);
    }

    public MorflowException(String message, Throwable cause) {
        super(message, cause);
    }
}
