import pygame
class DmControlRenderer():
    def __init__(self, physics):
        self.physics = physics
        self.screen = None

    def render(self, *args, width=480, height=480, camera_id=0, **kwargs):

        # safe for multiple calls
        pygame.init()
        if not self.screen:
            self.screen = pygame.display.set_mode((width, height))
        else:
            c_width, c_height = self.screen.get_size()
            if c_width != width or c_height != height:
                self.screen = pygame.display.set_mode((width, height))

        im = self.physics.render(width=width, height=height, camera_id=camera_id).transpose((1,0,2))
        pygame.pixelcopy.array_to_surface(self.screen, im)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

    def render_frame(self, *args, width=480, height=480, camera_id=0, **kwargs):
        return self.physics.render(width=width, height=height, camera_id=camera_id).transpose((1,0,2))
