import future
import pygame
import os
import random
import math


pygame.init()
pygame.font.init()
pygame.display.set_mode((1024, 1024))

class World:
    font_pathes = [os.path.join("fonts", f) for f in os.listdir("fonts") if f.endswith(".ttf")]
    #categories = ["can"]
    def __init__(self, w, h):
        self.w = w
        self.h = h
        # gen trashes randomly
        self.trashes = {
            "can": [Can() for _ in range(random.randrange(1, 10))]
        }
        for _, trashes in self.trashes.items():
            for trash in trashes:
                radius=(trash.sprite.get_width()**2 + trash.sprite.get_height()**2)**0.5*0.5
                trash.move_to(radius+random.random()*(w-radius*2), radius+random.random()*(h-radius*2))
                trash.rotate_to(random.random()*math.pi*2)
        # gen robot random position
        self.robot = Robot()
        radius = (self.robot.sprite.get_width()**2 + self.robot.sprite.get_height()**2)**0.5*0.5
        self.robot.move_to(radius+random.random()*(w-radius*2), radius+random.random()*(h-radius*2))
        self.robot.rotate_to(random.random()*math.pi*2)

        # gen background 
        # should be with noise(not currently support)
        self.background = pygame.Surface((w, h))
        self.background.fill((255, 255, 255))

        # gen category label randomly
        #self.fonts = [pygame.font.Font(random.choice(World.font_pathes), random.randrange(80, 100)) for _ in World.categories]
        #self.category_label_sprites = [self.fonts[i].render(World.categories[i], True, (0, 0, 0)) for i in range(len(World.categories))]
        #self.category_label_poses = [(random.random()*(w-cls.get_width()), random.random()*(h-cls.get_height())) for cls in self.category_label_sprites]
        
        # gen screen(which all of above are drawn)
        self.screen = pygame.Surface((w, h))
    def get_score(self):
        # get distacne score from each trashes
        distance_square_sum=0
        for _, trashes in self.trashes.items():
            for i in range(len(trashes)):
                for j in range(i+1, len(trashes)):
                    trash1 = trashes[i]
                    trash2 = trashes[j]
                    distance_square_sum += trash1.distance(trash2)**2
        return -distance_square_sum
        
        
            
    def get_screen(self):
        # draw background
        self.screen.blit(self.background, (0,0))
        # draw category label
        #for i in range(len(World.categories)):
        #    self.screen.blit(self.category_label_sprites[i], self.category_label_poses[i])
        # draw trash
        for _, trashes in self.trashes.items():
            for trash in trashes: 
                if trash == self.robot.grab_thing:
                    continue
                trash_sprite = pygame.transform.rotate(trash.sprite, trash.angle/math.pi*180)
                self.screen.blit(trash_sprite, (trash.x-trash_sprite.get_width()*0.5, trash.y-trash_sprite.get_height()*0.5))
        # draw grub thing if exist
        if self.robot.grab_thing is not None: 
            trash = self.robot.grab_thing
            trash_sprite = pygame.transform.rotate(trash.sprite, trash.angle/math.pi*180)
            self.screen.blit(trash_sprite, (trash.x-trash_sprite.get_width()*0.5, trash.y-trash_sprite.get_height()*0.5))
        # draw robot
        robot_sprite = pygame.transform.rotate(self.robot.sprite, self.robot.angle/math.pi*180)
        self.screen.blit(robot_sprite, (self.robot.x-robot_sprite.get_width()*0.5, self.robot.y-robot_sprite.get_height()*0.5))
        return self.screen
    def draw_on(self, dest):
        dest.blit(pygame.transform.scale(self.get_screen(), (dest.get_width(), dest.get_height())), (0,0))
        


class Thing:
    def __init__(self, name, w, h):
        self.name = name
        self.x = 0
        self.y = 0
        self.w = w
        self.h = h
        self.angle = 0
    def move_to(self, x, y):
        self.x = x
        self.y = y
    def move_by(self, x, y):
        self.x += x
        self.y += y
    def rotate_to(self, angle):
        self.angle = angle
    def rotate_by(self, angle):
        self.angle += angle
    def distance(self, other):
        return ((self.x-other.x)**2 + (self.x-other.x)**2)**0.5

class Can(Thing):
    image_pathes = [os.path.join("images/can", f) for f in os.listdir("images/can") if f.endswith(".png")]
    images = [pygame.image.load(path).convert_alpha() for path in image_pathes]
    sprites = [image for image in images]
    def __init__(self):
        self.sprite = random.choice(Can.sprites)
        Thing.__init__(self, "can", self.sprite.get_width(), self.sprite.get_height())
        
        
class Robot(Thing):
    arm_range = 100
    stop_image_path = "images/robot/stop.png"
    #grab_animation = []
    #move_animation= []
    stop_image = pygame.image.load(stop_image_path).convert_alpha()
    stop_sprite = stop_image
    wheel_speed = 1

    #grab_animation_surfaces = [pygame.image.load(i) for i in grab_animation_path]
    #move_animation_surfaces = [pygame.image.load(i) for i in move_animation_path]
    def __init__(self):
        Thing.__init__(self, "robot", Robot.stop_sprite.get_width(), Robot.stop_sprite.get_height())
        self.grab_thing = None
        self.wheel_speed = Robot.wheel_speed
        self.turn_speed = self.wheel_speed*2/((self.w**2+self.h**2)**0.5)
        self.sprite = Robot.stop_sprite
    def try_grab_nearlist(self, world):
        if self.grab_thing is not None: return False
        arm_pos = (self.x+math.cos(self.angle)*0.5*self.w, self.y-math.sin(self.angle)*0.5*self.w)
        best_d = 0
        for _, trashes in world.trashes.items():
            for thing in trashes:
                d2 = (thing.x - arm_pos[0])**2 + (thing.y - arm_pos[1])**2
                if d2 < Robot.arm_range**2:
                    if self.grab_thing is None:
                        best_d = d2 
                        self.grab_thing = thing
                    elif best_d >= d2:
                        self.grab_thing = thing
        if self.grab_thing is not None: 
            self.grab_thing.move_to(*arm_pos)
            return True
        else: return False
    def try_grab(self, thing):
        if self.grab_thing is not None: return False
        arm_pos = (self.x+math.cos(self.angle)*0.5*self.w, self.y-math.sin(self.angle)*0.5*self.w)
        if (thing.x - arm_pos[0])**2 + (thing.y - arm_pos[1])**2 < Robot.arm_range**2:
            #things are in range of arm(around circle of arm)
            self.grab_thing = thing
            self.grab_thing.move_to(*arm_pos)
            return True
        return False
    def put(self):
        self.grab_thing = None
    def move_to(self, x, y):
        dx = x - self.x
        dy = y - self.y
        self.move_by(dx, dy)
    def move_by(self, x, y):
        self.x += x
        self.y += y
        if self.grab_thing is not None:
            self.grab_thing.move_by(x, y)
    def rotate_to(self, angle):
        dangle = angle - self.angle
        self.rotate_by(dangle)
    def rotate_by(self, angle):
        self.angle += angle
        if self.grab_thing is not None:
            self.grab_thing.rotate_by(angle)
            arm_pos = (self.x+math.cos(self.angle)*0.5*self.w, self.y-math.sin(self.angle)*0.5*self.w)
            self.grab_thing.move_to(*arm_pos)

    def move_forward(self):
        self.move_by(math.cos(self.angle)*self.wheel_speed, -math.sin(self.angle)*self.wheel_speed)
    def move_backward(self):
        self.move_by(-math.cos(self.angle)*self.wheel_speed, math.sin(self.angle)*self.wheel_speed)
    def rotate_right(self):
        self.rotate_by(-self.turn_speed)
    def rotate_left(self):
        self.rotate_by(self.turn_speed)
    def wheel_move(self, left_top, right_top, left_bottom, right_bottom):
        # move vertical
        vertical_speed = self.wheel_speed*(left_top + right_top + left_bottom + right_bottom)*0.25
        self.move_by(math.cos(self.angle)*vertical_speed, -math.sin(self.angle)*vertical_speed)

        # move horizental
        horizental_speed = self.wheel_speed*(-left_top+right_top+left_bottom-right_bottom)*0.25
        self.move_by(math.cos(self.angle-math.pi*0.5)*horizental_speed, -math.sin(self.angle-math.pi*0.5)*horizental_speed)
        
