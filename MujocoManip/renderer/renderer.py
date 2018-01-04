import glfw

class Renderer():
    # https://github.com/FlorianRhiem/pyGLFW
    def __enter__(self):
        # Initialize the library
        if not glfw.init():
            raise RuntimeError('GLFW initialization failed')

        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(640, 480, "Hello World", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('GLFW window initialization failed')
        # Make the window's context current
        glfw.make_context_current(window)

    def render(self, physics):
        while not glfw.window_should_close(window):
            # Render here, e.g. using pyOpenGL
            

            # Swap front and back buffers
            glfw.swap_buffers(window)

            # Poll for and process events
            glfw.poll_events()



    def __exit__(self):
        glfw.terminate()


def main():
    
    
    
    window = 
    

    

    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # Render here, e.g. using pyOpenGL

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    

if __name__ == "__main__":
    main()