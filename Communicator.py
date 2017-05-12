import pygame
import sys
import math
from Thing import World, Thing, Robot, Can

def gen_world(width, height):
    world = World(width, height)
    return world

def init_world(world):
    world.__init__(world.w, world.h)

def send_action(world, action):
    if action == "forward":
        world.robot.move_forward()
    elif action == "turn_left":
        world.robot.rotate_left()
    elif action == "turn_right":
        world.robot.rotate_right()
    elif action == "grab":
        world.robot.try_grab_nearlist(world)
    elif action == "put":
        world.robot.put()
    else:
        print("undefined action for robot: " + action)
    
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
        if keys[pygame.K_UP]:
            send_action(world, "forward")
        if keys[pygame.K_LEFT]:
            send_action(world, "turn_left")
        if keys[pygame.K_RIGHT]:
            send_action(world, "turn_right")
        if keys[pygame.K_a]:
            send_action(world, "grab")
        if keys[pygame.K_s]:
            send_action(world, "put")
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
        pygame.display.set_caption(TITLE + "/FPS: "+ str(round(clock.get_fps())) + "/SCORE: " + str(world.get_score()))


world = gen_world(800, 800)
test(world, 800, 800)