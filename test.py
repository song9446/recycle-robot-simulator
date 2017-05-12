import pygame
import sys
import math

clock = pygame.time.Clock()

TITLE="Recycler"
WIDTH=800
HEIGHT=800

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

from Thing import World, Thing, Robot, Can

world = World(WIDTH,HEIGHT)

while True:
    world.draw_on(screen)
    pygame.display.flip()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        world.robot.move_forward()
    if keys[pygame.K_LEFT]:
        world.robot.rotate_left()
    if keys[pygame.K_RIGHT]:
        world.robot.rotate_right()
    if keys[pygame.K_a]:
        print(world.robot.try_grab_nearlist(world.trashes))
    if keys[pygame.K_s]:
        world.robot.put()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == pygame.LEFT:
            pass
        elif event.type == pygame.MOUSEBUTTONUP and event.button == pygameLEFT:
            pass
        elif event.type == pygame.MOUSEMOTION:
            pass
    clock.tick()
    pygame.display.set_caption(TITLE + " / FPS: "+ str(round(clock.get_fps())))

