# Communicator with Simmulator.
# Generally, You never cares about Simmulator.
# You can all task what you want with only Communicator.

import pygame
import sys
import math
from Simulator import World, Thing, Robot, Can

def gen_world(width, height):
    world = World(width, height)
    return world

def init_world(world):
    world.__init__(world.w, world.h)

world = None
ACTIONS= {
    "move_forward": lambda world, v: world.robot.move_forward(v),
    "move_backward": lambda world, v: world.robot.move_backward(v),
    "move_left": lambda world, v: world.robot.move_right(v),
    "move_right": lambda world, v: world.robot.move_left(v),
    "turn_left": lambda world, w: world.robot.rotate_left(w),
    "turn_right": lambda world, w: world.robot.rotate_right(w),
    "grab": lambda world: world.robot.try_grab_nearlist(world),
    "put": lambda world: world.robot.put(),
    "wheel_move": lambda world, lt, rt, lb, rb: world.robot.wheel_move(lt, rt, lb, rb),
    "move": lambda world, x, y, r: world.robot.move(x, y, r),
    "absolute_move": lambda world, x, y, w: world.robot.absolute_move(x, y, w),
}

def send_action(world, action, *args, **kwargs):
    ACTIONS[action](world, *args, **kwargs)
    
def get_reward(world):
    return world.get_score()

def get_screen_pixels(world, w, h):
    return pygame.surfarray.array2d(pygame.transform.scale(world.get_screen(), (w, h))).view('uint8').reshape((w, h, 4,))[..., :3][:,:,::-1]






clock = pygame.time.Clock()
TITLE="Recycler"
pygame.init()
screen = pygame.display.set_mode((0, 0))
def show(world, w=0, h=0):
    if (w==0 or h==0) and (w != world.w or h != world.h):
        w = world.w
        h = world.h
    if show.last_w != w or show.last_h != h:
        screen = pygame.display.set_mode((w, h))
        last_w, last_h = w, h
    world.draw_on(screen)
    pygame.display.flip()
    clock.tick()
    #pygame.display.set_caption(TITLE + "/FPS: "+ str(round(clock.get_fps())) + "/SCORE: " + str(get_reward(world)))
show.last_w = 0
show.last_h = 0




def test(world, w=0, h=0):
    '''
    This function is just for the test.
    You can see the gui and you can test the action works well by a keyboard.
    '''
    while True:
        show(world)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            send_action(world, "move_forward", 1)
        if keys[pygame.K_a]:
            send_action(world, "turn_left", 1)
        if keys[pygame.K_d]:
            send_action(world, "turn_right", 1)
        if keys[pygame.K_s]:
            send_action(world, "move_backward", 1)
        if keys[pygame.K_UP]:
            send_action(world, "grab")
        if keys[pygame.K_DOWN]:
            send_action(world, "put")
        if keys[pygame.K_q]:
            send_action(world, "move_left", 1)
            #send_action(world, "wheel_move", 1, -1, -1, 1)
        if keys[pygame.K_e]:
            send_action(world, "move_right", 1)
            #send_action(world, "wheel_move", -1, 1, 1, -1)
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


if __name__ == '__main__':
    world = gen_world(800, 800)
    test(world, 300, 300)
