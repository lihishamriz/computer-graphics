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
        return np.inf

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
            intersect = obj.intersect(self)
            if intersect and intersect[0] and intersect[0] < min_distance:
                nearest_object = intersect[1]
                min_distance = intersect[0]
        
        return min_distance, nearest_object


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
        
    def compute_normal(self, hit=None):
        return self.normal

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

    def compute_normal(self, hit=None):
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
        g = np.subtract(f, (np.subtract(d, a)))
        h = np.subtract(e, (np.subtract(c, b)))
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

    def compute_normal(self, hit):
        return normalize(hit - self.center)

    def intersect(self, ray: Ray):
        center_to_origin = ray.origin - self.center
        b = np.dot(ray.direction, center_to_origin)
        delta = b ** 2 - (np.linalg.norm(center_to_origin) ** 2 - self.radius ** 2)
        if delta >= 0:
            t1 = -b + (delta ** 0.5)
            t2 = -b - (delta ** 0.5)
            t = np.minimum(t1, t2)
            if t > 0:
                return t, self
        return None
    
    
class Scene:
    def __init__(self, ambient, lights, objects):
        self.ambient = ambient
        self.lights = lights
        self.objects = objects

    def calc_ambient_color(self, obj):
        return self.ambient * obj.ambient

    @staticmethod
    def calc_diffuse_color(obj, hit, light):
        K_D = obj.diffuse
        I_L = light.get_intensity(hit)
        N = obj.compute_normal(hit)
        L = normalize(light.get_light_ray(hit).direction)
        return K_D * I_L * np.dot(N, L)

    @staticmethod
    def calc_specular_color(obj, ray, hit, light):
        K_S = obj.specular
        I_L = light.get_intensity(hit)
        V = normalize(ray.origin - hit)
        L = normalize(light.get_light_ray(hit).direction)
        R = reflected(L, obj.compute_normal(hit))
        return K_S * I_L * (np.dot(V, R) ** (obj.shininess / 10))
    
    def calc_shadow_factor(self, hit, light):
        light_ray = light.get_light_ray(hit)
        intersection = light_ray.nearest_intersected_object(self.objects)
        if intersection and intersection[0] and intersection[0] < light.get_distance_from_light(hit):
            return 0
        return 1

    @staticmethod
    def construct_reflective_ray(obj, ray, hit):
        reflected_ray = reflected(ray.direction, obj.compute_normal(hit))
        new_ray = Ray(hit, reflected_ray)
        return new_ray
    
    def find_intersection(self, ray):
        t, nearest_obj = ray.nearest_intersected_object(self.objects)
        if nearest_obj:
            hit = ray.origin + t * ray.direction
            return hit, nearest_obj
        return np.inf, None

    def get_color(self, obj, ray, hit, level, max_level):
        normal = obj.compute_normal(hit)
        hit += 0.01 * normal
        color = self.calc_ambient_color(obj)
        for light in self.lights:
            if self.calc_shadow_factor(hit, light):
                color = color + self.calc_diffuse_color(obj, hit, light) + self.calc_specular_color(obj, ray, hit, light)
        level += 1
        if level > max_level:
            return color
        r_ray = self.construct_reflective_ray(obj, ray, hit)
        r_hit, nearest_obj = self.find_intersection(r_ray)
        if nearest_obj:
            color = color + nearest_obj.reflection * self.get_color(nearest_obj, r_ray, r_hit, level, max_level)
        return color
