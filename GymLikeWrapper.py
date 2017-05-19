class gymlike:
    def __init__(self):
        self.world = gen_world(500, 500)
    def reset(self):
        init_world(self.world)
    def render():
        show(self.world)
        
def gen_world(width, height):
