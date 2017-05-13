import pygame
import sys
import math
from Thing import World, Thing, Robot, Can

def gen_world(width, height):
    world = World(width, height)
    return world

def init_world(world):
    world.__init__(world.w, world.h)

ACTIONS= {
    "move_forward": lambda: world.robot.move_forward(),
    "move_backward": lambda: world.robot.move_backward(),
    "turn_left": lambda: world.robot.rotate_left(),
    "turn_right": lambda: world.robot.rotate_right(),
    "grab": lambda: world.robot.try_grab_nearlist(world),
    "put": lambda: world.robot.put(),
    "wheel_move": lambda lt, rt, lb, rb: world.robot.wheel_move(lt, rt, lb, rb)
}

def send_action(world, action, *args, **kwargs):
    ACTIONS[action](*args, **kwargs)
    
def get_reward(world):
    return world.get_score()
    

def get_screen_pixels(world, w, h):
    return pygame.surfarray.array2d(pygame.transform.scale(world.get_screen(), (w, h))).view('uint8').reshape((w, h, 4,))[..., :3][:,:,::-1]

def test(world, w=0, h=0):
    clock = pygame.time.Clock()

    TITLE="Recycler"

    pygame.init()
    if w==0: w=world.w
    if h==0: h=world.h
    screen = pygame.display.set_mode((w, h))


    while True:
        world.draw_on(screen)
        pygame.display.flip()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            send_action(world, "move_forward")
        if keys[pygame.K_a]:
            send_action(world, "turn_left")
        if keys[pygame.K_d]:
            send_action(world, "turn_right")
        if keys[pygame.K_s]:
            send_action(world, "move_backward")
        if keys[pygame.K_UP]:
            send_action(world, "grab")
        if keys[pygame.K_DOWN]:
            send_action(world, "put")
        if keys[pygame.K_q]:
            send_action(world, "wheel_move", 1, -1, -1, 1)
        if keys[pygame.K_e]:
            send_action(world, "wheel_move", -1, 1, 1, -1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pass
            elif event.type == pygame.MOUSEBUTTONUP:
                pass
            elif event.type == pygame.MOUSEMOTION:
                pass
        clock.tick()
        pygame.display.set_caption(TITLE + "/FPS: "+ str(round(clock.get_fps())) + "/SCORE: " + str(get_reward(world)))


world = gen_world(800, 800)
test(world, 800, 800)
