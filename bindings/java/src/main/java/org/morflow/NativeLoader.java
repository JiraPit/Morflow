package org.morflow;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Utility to discover, unpack, and load the Morflow JNI native shared library.
 */
public final class NativeLoader {
    private static boolean loaded = false;

    private NativeLoader() {}

    public static synchronized void load() {
        if (loaded) return;

        // 1. Try standard System.loadLibrary (if present in java.library.path)
        try {
            System.loadLibrary("morflow_jni");
            loaded = true;
            return;
        } catch (UnsatisfiedLinkError ignored) {}

        // 2. Search development build paths
        String os = System.getProperty("os.name").toLowerCase();
        String libFileName;
        if (os.contains("win")) {
            libFileName = "morflow_jni.dll";
        } else if (os.contains("mac")) {
            libFileName = "libmorflow_jni.dylib";
        } else {
            libFileName = "libmorflow_jni.so";
        }

        String[] devPaths = {
            "target/release/" + libFileName,
            "target/debug/" + libFileName,
            "../target/release/" + libFileName,
            "../target/debug/" + libFileName,
            "../../target/release/" + libFileName,
            "../../target/debug/" + libFileName,
            "../../../target/release/" + libFileName,
            "../../../target/debug/" + libFileName,
            "../../../../target/release/" + libFileName,
            "../../../../target/debug/" + libFileName,
        };

        for (String candidate : devPaths) {
            File f = new File(candidate);
            if (f.exists() && f.isFile()) {
                try {
                    System.load(f.getAbsolutePath());
                    loaded = true;
                    return;
                } catch (UnsatisfiedLinkError ignored) {}
            }
        }

        // 3. Try unpacking from JAR resources
        String arch = System.getProperty("os.arch").toLowerCase();
        String resourceDir;
        if (os.contains("win")) {
            resourceDir = "windows-" + (arch.contains("64") ? "x64" : "x86");
        } else if (os.contains("mac")) {
            resourceDir = "macos-" + (arch.contains("aarch64") || arch.contains("arm") ? "aarch64" : "x64");
        } else {
            resourceDir = "linux-" + (arch.contains("aarch64") || arch.contains("arm") ? "aarch64" : "x64");
        }

        String resourcePath = "/native/" + resourceDir + "/" + libFileName;
        try (InputStream is = NativeLoader.class.getResourceAsStream(resourcePath)) {
            if (is != null) {
                Path tempDir = Files.createTempDirectory("morflow_jni_");
                File tempFile = tempDir.resolve(libFileName).toFile();
                tempFile.deleteOnExit();
                try (FileOutputStream fos = new FileOutputStream(tempFile)) {
                    is.transferTo(fos);
                }
                System.load(tempFile.getAbsolutePath());
                loaded = true;
                return;
            }
        } catch (IOException ignored) {}

        throw new UnsatisfiedLinkError("Could not load native library 'morflow_jni' from java.library.path, dev paths, or JAR resources.");
    }
}
