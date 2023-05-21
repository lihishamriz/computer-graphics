from helper_classes import *
import matplotlib.pyplot as plt


def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom
    
    image = np.zeros((height, width, 3))
    scene = Scene(ambient, lights, objects)
    
    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)
            
            color = np.zeros(3)
            
            # This is the main loop where each pixel color is computed.
            hit, nearest_obj = scene.find_intersection(ray)
            if nearest_obj:
                color = scene.get_color(nearest_obj, ray, hit, 1, max_depth)
            
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)
    
    return image


# Write your own objects and lights
def your_own_scene():
    camera = np.array([0, 0, 1])
    ambient = np.array([0.1, 0.1, 0.1])
    
    s_light = SpotLight(intensity=np.array([1, 1, 1]), position=np.array([0, 6, -10]), direction=([0, -1, 1]),
                        kc=0.1, kl=0.1, kq=0.1)
    d_light = DirectionalLight(intensity=np.array([1, 1, 1]), direction=np.array([1, 1, 1]))
    lights = [s_light, d_light]
    
    background = Plane([0, 0, 1], [0, 0, -30])
    background.set_material([0.2, 1, 1], [0.2, 1, 1], [0, 0, 0], 100, 0)
    
    plane = Plane([0, 1, 0], [0, -1, 0])
    plane.set_material([0.5, 0.9, 0.5], [0.5, 0.9, 0.5], [0, 0, 0], 1000, 0.5)
    
    sphere_a = Sphere([0, 0, -2], 1)
    sphere_a.set_material([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1, 1, 1], 1000, 0, 0.5, 1)
    
    sphere_b = Sphere([-1, -0.5, -8], 0.5)
    sphere_b.set_material([1, 0, 1], [1, 0, 1], [0, 0, 0], 1000, 0)
    
    sphere_c = Sphere([1, -0.5, -8], 0.5)
    sphere_c.set_material([1, 0, 1], [1, 0, 1], [0, 0, 0], 1000, 0)
    
    sphere_d = Sphere([0, -0.7, -4], 0.3)
    sphere_d.set_material([1, 0, 1], [1, 0, 1], [0, 0, 0], 1000, 0)
    
    rectangle = Rectangle([-2.5, 10, -12], [-2.5, 6, -15], [2.5, 6, -15], [2.5, 10, -12])
    rectangle.set_material([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0, 0, 0], 100, 0.8)
    
    objects = [background, plane, sphere_a, sphere_b, sphere_c, sphere_d, rectangle]
    
    return camera, ambient, lights, objects
