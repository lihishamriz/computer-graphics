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
# rectangle = Rectangle([0,1,-1],[0,-1,-1],[2,-1,-2],[2,1,-2])
# rectangle.set_material([1, 0, 0], [1, 0, 0], [0, 0, 0], 100, 0.5)
# plane = Plane([0,0,1], [0,0,-3])
# plane.set_material([0, 0.5, 0], [0, 1, 0], [1, 1, 1], 100, 0.5)
#
#
# objects = [rectangle,plane]
#
# light = DirectionalLight(intensity= np.array([1, 1, 1]),direction=np.array([1,1,1]))
#
# lights = [light]
#
# ambient = np.array([0.1,0.1,0.1])
#
# camera = np.array([0,0,1])
#
# im = render_scene(camera, ambient, lights, objects, (64,64), 1)
# plt.imshow(im)
# plt.imsave('scene2.png', im)


# Scene 3
# sphere_a = Sphere([-0.5, 0.2, -1],0.5)
# sphere_a.set_material([1, 0, 0], [1, 0, 0], [0.3, 0.3, 0.3], 100, 1)
#
# cuboid = Cuboid(
#     [-1, -.75, -2],
#     [-1,-2, -2],
#     [ 1,-2, -1.5],
#     [ 1, -.75, -1.5],
#     [ 2,-2, -2.5],
#     [ 2, -.75, -2.5]
#     )
#
#
# cuboid.set_material([1, 0, 0], [1, 0, 0], [0, 0, 0], 100, 0.5)
# cuboid.apply_materials_to_faces()
#
# sphere_b = Sphere([0.8, 0, -0.5],0.3)
# sphere_b.set_material([0, 1, 0], [0, 1, 0], [0.3, 0.3, 0.3], 100, 0.2)
# plane = Plane([0,1,0], [0,-2,0])
# plane.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1, 1, 1], 1000, 0.5)
#
# background = Plane([0,0,1], [0,0,-10])
# background.set_material([0, 0.5, 0], [0, 1, 0], [1, 1, 1], 100, 0.5)
#
# objects = [cuboid,plane,background]
#
# light = PointLight(intensity= np.array([1, 1, 1]),position=np.array([2,2,1]),kc=0.1,kl=0.1,kq=0.1)
#
# lights = [light]
#
# ambient = np.array([0.1,0.2,0.3])
#
# camera = np.array([0,0,1])
#
# im = render_scene(camera, ambient, lights, objects, (256,256), 3)
# plt.imshow(im)
# plt.imsave('scene3.png', im)


# Scene 4
# sphere_a = Sphere([-0.5, 0.2, -1],0.5)
# sphere_a.set_material([1, 0, 0], [1, 0, 0], [0.3, 0.3, 0.3], 100, 1)
# sphere_b = Sphere([0.8, 0, -0.5],0.3)
# sphere_b.set_material([0, 1, 0], [0, 1, 0], [0.3, 0.3, 0.3], 100, 0.2)
# plane = Plane([0,1,0], [0,-0.3,0])
# plane.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1, 1, 1], 1000, 0.5)
# background = Plane([0,0,1], [0,0,-3])
# background.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], 1000, 0.5)
#
#
# objects = [sphere_a,sphere_b,plane,background]
#
# light = PointLight(intensity= np.array([1, 1, 1]),position=np.array([1,1.5,1]),kc=0.1,kl=0.1,kq=0.1)
#
# lights = [light]
#
# ambient = np.array([0.1,0.2,0.3])
#
# camera = np.array([0,0,1])
#
# im = render_scene(camera, ambient, lights, objects, (256,256), 3)
# plt.imshow(im)
# plt.imsave('scene4.png', im)


# Scene 5
# background = Plane([0,0,1], [0,0,-1])
# background.set_material([1, 1, 1], [1, 1, 1], [1, 1, 1], 1000, 0.5)
#
#
# objects = [background]
#
# light_a = SpotLight(intensity= np.array([0, 0, 1]),position=np.array([0.5,0.5,0]), direction=([0,0,1]),
#                     kc=0.1,kl=0.1,kq=0.1)
# light_b = SpotLight(intensity= np.array([0, 1, 0]),position=np.array([-0.5,0.5,0]), direction=([0,0,1]),
#                     kc=0.1,kl=0.1,kq=0.1)
# light_c = SpotLight(intensity= np.array([1, 0, 0]),position=np.array([0,-0.5,0]), direction=([0,0,1]),
#                     kc=0.1,kl=0.1,kq=0.1)
#
# lights = [light_a,light_b,light_c]
#
# ambient = np.array([0,0,0])
#
# camera = np.array([0,0,1])
#
# im = render_scene(camera, ambient, lights, objects, (256,256), 3)
# plt.imshow(im)
# plt.imsave('scene5.png', im)

# Bonus
# from hw3 import *
#
#
# plane1 = Plane([0,0,1], [0,0,-3])
# plane1.set_material([0, 1, 0], [0, 1, 0], [0, 0, 0], 1000, 0)
#
# sphere_a = Sphere([0, 0, -1],1)
# sphere_a.set_material([1, 0, 0], [1, 0, 0], [0.3, 0.3, 0.3], 100, 0, 0.5, 1.458)
#
# objects = [plane1,sphere_a]
#
# light = PointLight(intensity= np.array([1, 1, 1]),position=np.array([1,1.5,1]),kc=0.1,kl=0.1,kq=0.1)
#
# lights = [light]
#
# ambient = np.array([0.1,0.2,0.3])
#
# camera = np.array([0,0,1])
#
# im = render_scene(camera, ambient, lights, objects, (256,256), 3)
# plt.imshow(im)
# plt.imsave('scene4.png', im)


# Scene 6
camera, ambient, lights, objects = your_own_scene()

im = render_scene(camera, ambient, lights, objects, (256,256), 3)
plt.imshow(im)
plt.imsave('scene6.png', im)