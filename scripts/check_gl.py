import ctypes
import os

import glfw


def gl_string(lib, name):
    value = lib.glGetString(name)
    return value.decode("utf-8") if value else None


def main():
    print("DISPLAY:", os.environ.get("DISPLAY"))
    print("LIBGL_ALWAYS_INDIRECT:", os.environ.get("LIBGL_ALWAYS_INDIRECT"))
    print("LIBGL_ALWAYS_SOFTWARE:", os.environ.get("LIBGL_ALWAYS_SOFTWARE"))
    print("MESA_GL_VERSION_OVERRIDE:", os.environ.get("MESA_GL_VERSION_OVERRIDE"))

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(320, 240, "gl-diagnostic", None, None)
    print("window_created:", bool(window))
    if not window:
        glfw.terminate()
        raise RuntimeError("glfw.create_window() failed")

    glfw.make_context_current(window)
    lib = ctypes.CDLL("libGL.so.1")
    lib.glGetString.restype = ctypes.c_char_p

    print("GL_VENDOR:", gl_string(lib, 0x1F00))
    print("GL_RENDERER:", gl_string(lib, 0x1F01))
    print("GL_VERSION:", gl_string(lib, 0x1F02))

    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
