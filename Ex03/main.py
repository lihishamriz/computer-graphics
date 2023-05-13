from hw3 import *

# Scene 1
# plane_a = Plane([0,1,0],[0,-1,0])
# plane_a.set_material([0.3, 0.5, 1], [0.3, 0.5, 1], [1, 1, 1], 10, 0.5)
# plane_b = Plane([0,0,1], [0,0,-3])
# plane_b.set_material([0, 0.5, 0], [0, 1, 0], [1, 1, 1], 10, 0.5)
#
#
# objects = [plane_a, plane_b]
#
# light = PointLight(intensity= np.array([1, 1, 1]),position=np.array([1,1,1]),kc=0.1,kl=0.1,kq=0.1)
#
# lights = [light]
#
# ambient = np.array([0.1,0.1,0.1])
#
# camera = np.array([0,0,1])
#
# im = render_scene(camera, ambient, lights, objects, (256,256), 1)
# plt.imshow(im)
# plt.imsave('scene1.png', im)


# Scene 2
rectangle = Rectangle([0,1,-1],[0,-1,-1],[2,-1,-2],[2,1,-2])
rectangle.set_material([1, 0, 0], [1, 0, 0], [0, 0, 0], 100, 0.5)
plane = Plane([0,0,1], [0,0,-3])
plane.set_material([0, 0.5, 0], [0, 1, 0], [1, 1, 1], 100, 0.5)


objects = [rectangle,plane]

light = DirectionalLight(intensity= np.array([1, 1, 1]),direction=np.array([1,1,1]))

lights = [light]

ambient = np.array([0.1,0.1,0.1])

camera = np.array([0,0,1])

im = render_scene(camera, ambient, lights, objects, (64,64), 1)
plt.imshow(im)
plt.imsave('scene2.png', im)