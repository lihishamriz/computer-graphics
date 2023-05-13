import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    # v = np.array([0, 0, 0])
    v = vector - 2 * np.dot(vector, axis) * axis
    return v

## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):
    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = direction

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection_point):
        return Ray(intersection_point, self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return 1

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = direction
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        v_normalized = normalize(self.position - intersection)
        return self.intensity * np.dot(v_normalized, self.direction) / (self.kc + self.kl*d + self.kq * (d**2))


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        intersections = None
        nearest_object = None
        min_distance = np.inf
        
        for obj in objects:
            ray = Ray(self.origin, self.direction)
            intersect = obj.intersect(ray)
            if intersect and intersect[0] and intersect[0] < min_distance:
                nearest_object = obj
                min_distance = intersect[0]
        
        return nearest_object, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        dot = np.dot(self.normal, ray.direction)
        if dot:
            t = (np.dot(v, self.normal) / dot)
            if t > 0:
                return t, self
        else:
            return None


class Rectangle(Object3D):
    """
        A rectangle is defined by a list of vertices as follows:
        a _ _ _ _ _ _ _ _ d
         |               |  
         |               |  
         |_ _ _ _ _ _ _ _|
        b                 c
        This function gets the vertices and creates a rectangle object
    """
    def __init__(self, a, b, c, d):
        """
            ul -> bl -> br -> ur
        """
        self.abcd = [np.asarray(v) for v in [a, b, c, d]]
        self.normal = self.compute_normal()

    def compute_normal(self):
        v1 = self.abcd[1] - self.abcd[0]
        v2 = self.abcd[3] - self.abcd[0]
        n = normalize(np.cross(v1, v2))
        return n

    # Intersect returns both distance and nearest object.
    # Keep track of both.
    def intersect(self, ray: Ray):
        plane = Plane(self.normal, self.abcd[0])
        intersection = plane.intersect(ray)
        if not intersection:
            return None
        if intersection[0] > 0:
            point = ray.origin + intersection[0] * ray.direction
            for i in range(4):
                p1 = self.abcd[i] - point
                p2 = self.abcd[(i+1) % 4] - point
                if np.dot(np.cross(p1, p2), self.normal) <= 0:
                    return None
            return intersection[0], self

        return None

class Cuboid(Object3D):
    def __init__(self, a, b, c, d, e, f):
        """ 
              g+---------+f
              /|        /|
             / |  E C  / |
           a+--|------+d |
            |Dh+------|B +e
            | /  A    | /
            |/     F  |/
           b+--------+/c
        """
        g = [a[0], a[1], f[2]]
        h = [b[0], b[1], e[2]]
        # A = B = C = D = E = F = None
        A = Rectangle(a, b, c, d)
        B = Rectangle(d, c, e, f)
        C = Rectangle(g, h, e, f)
        D = Rectangle(a, b, h, g)
        E = Rectangle(g, a, d, f)
        F = Rectangle(h, b, c, e)
        self.face_list = [A, B, C, D, E, F]

    def apply_materials_to_faces(self):
        for t in self.face_list:
            t.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)

    # Hint: Intersect returns both distance and nearest object.
    # Keep track of both
    def intersect(self, ray: Ray):
        return ray.nearest_intersected_object(self.face_list)
    

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        #TODO
        pass
    
class Scene:
    def __init__(self, camera, ambient, lights, objects):
        self.camera = camera
        self.ambient = ambient
        self.lights = lights
        self.objects = objects
        self.nearest_intersected_object = None

    def calc_ambient_color(self):
        return self.ambient * self.nearest_intersected_object.ambient

    def calc_diffuse_color(self, hit, ray: Ray, light):
        K_D = self.nearest_intersected_object.diffuse
        I_L = light.get_intensity(hit)
        N = self.nearest_intersected_object.normal
        L = normalize(light.get_distance_from_light(hit) * light.get_light_ray(hit).direction)
        return K_D * I_L * np.dot(N, L)

    def calc_specular_color(self, hit, ray, light):
        K_S = self.nearest_intersected_object.specular
        I_L = light.get_intensity(hit)
        V = normalize(ray.origin - hit)
        L = normalize(light.get_distance_from_light(hit) * light.get_light_ray(hit).direction)
        R = reflected(L, self.nearest_intersected_object.normal)
        return K_S * I_L * np.dot(V, R) ** self.nearest_intersected_object.shininess

    def calc_shadow_factor(self, hit, light):
        ray = Ray(hit, light.get_light_ray(hit).direction)
        for obj in self.objects:
            if obj == self.nearest_intersected_object:
                return 1
            intersection = obj.intersect(ray)
            if intersection and intersection[0] and intersection[0] < np.linalg.norm(hit - self.camera):
                return 0
        return 1

    def get_color(self, ray, hit, level, max_level):
        color = self.calc_ambient_color()
        for light in self.lights:
            s = self.calc_shadow_factor(hit, light)
            # s = 1
            color += (self.calc_diffuse_color(hit, ray, light) + self.calc_specular_color(hit, ray, light)) * s
        return color
        # level += 1
        # if level > max_level:
        #     return color
