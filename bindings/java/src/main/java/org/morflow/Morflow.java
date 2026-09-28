package org.morflow;

import java.nio.file.Path;

/**
 * Main entry point for compiling and loading Morflow data processing pipelines in Java.
 */
public final class Morflow {
    static {
        NativeLoader.load();
    }

    private Morflow() {}

    /**
     * Compiles and loads a `.morf` pipeline file from the given file path.
     *
     * @param path Path to the `.morf` pipeline file.
     * @return An executable Pipeline instance.
     */
    public static Pipeline load(String path) {
        return Pipeline.load(path);
    }

    /**
     * Compiles and loads a `.morf` pipeline file from the given Path.
     *
     * @param path Path to the `.morf` pipeline file.
     * @return An executable Pipeline instance.
     */
    public static Pipeline load(Path path) {
        return Pipeline.load(path.toAbsolutePath().toString());
    }

    /**
     * Compiles a `.morf` pipeline DSL source string directly.
     *
     * @param source Raw `.morf` pipeline DSL string.
     * @return An executable Pipeline instance.
     */
    public static Pipeline fromStr(String source) {
        return Pipeline.fromStr(source);
    }

    /**
     * Alias for {@link #fromStr(String)}.
     */
    public static Pipeline fromString(String source) {
        return Pipeline.fromStr(source);
    }
}
