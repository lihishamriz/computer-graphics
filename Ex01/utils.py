import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod


class SeamImage:
    def __init__(self, img_path, vis_seams=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            method (str) (a or b): a for Hard Vertical and b for the known Seam Carving algorithm
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path
        
        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T
        
        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()
        
        self.h, self.w = self.rgb.shape[:2]
        
        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
            self.mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices 
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))

    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        gs = np.dot(np_img, self.gs_weights)
        # Pad the boundaries with 0.5
        gs[[0, -1], :] = 0.5
        gs[:, [0, -1]] = 0.5

        return gs.astype('float32').reshape(self.h, self.w)

    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            In order to calculate a gradient of a pixel, only its neighborhood is required.
        """
        horizontal = np.abs(self.resized_gs - np.roll(self.resized_gs, -1, axis=0))
        vertical = np.abs(self.resized_gs - np.roll(self.resized_gs, -1, axis=1))
        # Handle last row and last column
        horizontal[-1, :] = horizontal[-2, :]
        vertical[:, -1] = vertical[:, -2]
        
        gradient = np.sqrt(horizontal ** 2 + vertical ** 2)
        gradient[gradient > 1] = 1

        return gradient

    def calc_M(self):
        pass
        
    def seams_removal(self, num_remove):
        pass

    def seams_removal_horizontal(self, num_remove):
        self.rotate_mats(-1)
        self.seams_removal(num_remove)
        self.rotate_mats(1)

    def seams_removal_vertical(self, num_remove):
        self.seams_removal(num_remove)

    def rotate_mats(self, clockwise):
        self.resized_gs = np.rot90(self.resized_gs, clockwise)
        self.resized_rgb = np.rot90(self.resized_rgb, clockwise)
        self.idx_map_h = np.rot90(self.idx_map_h, clockwise)
        self.idx_map_v = np.rot90(self.idx_map_v, clockwise)
        self.h, self.w = self.w, self.h

    def init_mats(self):
        pass

    def update_ref_mat(self):
        self.idx_map_v = self.idx_map_v[self.mask].reshape(self.h, self.w)
        self.idx_map_h = self.idx_map_h[self.mask].reshape(self.h, self.w)
        self.mask = self.mask[self.mask].reshape(self.h, self.w)

    def backtrack_seam(self):
        pass

    def remove_seam(self):
        self.w = self.w - 1
        three_d_mask = np.stack([self.mask] * 3, axis=2)
        self.resized_gs = self.resized_gs[self.mask].reshape(self.h, self.w)
        self.resized_rgb = self.resized_rgb[three_d_mask].reshape(self.h, self.w, 3)

    def reinit(self):
        """ re-initiates instance
        """
        self.__init__(self.path)

    @staticmethod
    def load_image(img_path):
        return np.asarray(Image.open(img_path)).astype('float32') / 255.0


class ColumnSeamImage(SeamImage):
    """ Column SeamImage.
    This class stores and implements all required data and algorithmics from implementing the "column" version of the seam carving algorithm.
    """
    def __init__(self, *args, **kwargs):
        """ ColumnSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def calc_M(self):
        """ Calculates the matrix M discussed in lecture, but with the additional constraint:
            - A seam must be a column. That is, the set of seams S is simply columns of M. 
            - implement forward-looking cost

        Returns:
            A "column" energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            The formula of calculation M is as taught, but with certain terms omitted.
            You might find the function 'np.roll' useful.
        """
        roll_left = np.roll(self.resized_gs, -1, axis=1)
        roll_right = np.roll(self.resized_gs, 1, axis=1)
        c_v = np.abs(roll_left - roll_right)
        M = self.E + c_v
        # Handle first and last columns - no c_v
        M[:, 0] = self.E[:, 0]
        M[:, -1] = self.E[:, -1]
        
        M = np.cumsum(M, axis=0)

        return M

    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams
        
        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matric
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) seam backtracking: calculates the actual indices of the seam
            iii) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            iv) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to support:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """
        self.E = self.calc_gradient_magnitude()
        self.M = self.calc_M()
        self.mask = np.ones_like(self.M, dtype=bool)
        for i in range(num_remove):
            min_seam_idx = np.argmin(self.M[-1])
            self.backtrack_seam()
            self.remove_seam()
            self.update_E(min_seam_idx)
            self.update_M(min_seam_idx)
            self.update_ref_mat()
        self.seams_rgb[~self.cumm_mask] = (1, 0, 0)

    def update_E(self, seam_idx):
        self.E = np.delete(self.E, seam_idx, axis=1)
        
        if seam_idx > 0:  # No need to handle first column
            col = self.resized_gs[:, seam_idx - 1]
            horizontal = np.abs(col - np.roll(col, -1))
            horizontal[-1] = horizontal[-2]
            if seam_idx == self.w:  # Handle last column
                vertical = self.E[:, self.w - 1]
            else:
                vertical = np.abs(col - self.resized_gs[:, seam_idx])
            gradient = np.sqrt(horizontal ** 2 + vertical ** 2)
            gradient[gradient > 1] = 1
            self.E[:, seam_idx - 1] = gradient

    def update_M(self, seam_idx):
        self.M = np.delete(self.M, seam_idx, axis=1)
        
        roll_left = np.roll(self.resized_gs, -1, axis=1)
        roll_right = np.roll(self.resized_gs, 1, axis=1)
        c_v = np.abs(roll_left - roll_right)
        # Handle first and last columns separately
        if seam_idx == 0:
            self.M[:, 0] = self.E[:, 0]
        elif seam_idx == self.w:
            self.M[:, -1] = self.E[:, -1]
        else:
            self.M[:, seam_idx - 1] = self.E[:, seam_idx - 1] + c_v[:, seam_idx - 1]
            self.M[:, seam_idx] = self.E[:, seam_idx] + c_v[:, seam_idx]
        self.M[:, seam_idx - 1:seam_idx + 1] = np.cumsum(self.M[:, seam_idx - 1:seam_idx + 1], axis=0)

    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        super().seams_removal_horizontal(num_remove)

    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of vertical seam to be removed
        """
        super().seams_removal_vertical(num_remove)

    def backtrack_seam(self):
        """ Backtracks a seam for Column Seam Carving method
        """
        min_seam_idx = np.argmin(self.M[-1])
        self.mask[:, min_seam_idx] = False
        self.cumm_mask[self.idx_map_v[:, min_seam_idx], self.idx_map_h[:, min_seam_idx]] = False

    def remove_seam(self):
        """ Removes a seam for self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """
        super().remove_seam()


class VerticalSeamImage(SeamImage):
    def __init__(self, *args, **kwargs):
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        M = np.zeros_like(self.resized_gs, dtype='float32')
        gs_roll_left = np.roll(self.resized_gs, -1, axis=1)
        gs_roll_right = np.roll(self.resized_gs, 1, axis=1)
        gs_roll_down = np.roll(self.resized_gs, 1, axis=0)
        c_v = np.abs(gs_roll_left - gs_roll_right)
        c_l = np.abs(gs_roll_left - gs_roll_right) + np.abs(gs_roll_down - gs_roll_right)
        c_r = np.abs(gs_roll_left - gs_roll_right) + np.abs(gs_roll_down - gs_roll_left)
        M[0] = self.E[0]
        for row in range(1, self.h):
            vertical = M[row - 1] + c_v[row]
            left = np.roll(M, 1, axis=1)[row - 1] + c_l[row]
            right = np.roll(M, -1, axis=1)[row - 1] + c_r[row]
            M[row] = self.E[row] + np.minimum(vertical, np.minimum(left, right))
            # Handle first and last columns - remove c_v
            M[row, 0] = self.E[row, 0] + min(vertical[0], right[0]) - c_v[row, 0]
            M[row, self.w - 1] = self.E[row, self.w - 1] + min(
                vertical[self.w - 1], left[self.w - 1]) - c_v[row, self.w - 1]
        return M

    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams
        
        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to supprt:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """
        self.E = self.calc_gradient_magnitude()
        self.M = self.calc_M()
        self.mask = np.ones_like(self.M, dtype=bool)
        for i in range(num_remove):
            self.backtrack_seam()
            self.remove_seam()
            self.E = self.calc_gradient_magnitude()
            self.M = self.calc_M()
            self.update_ref_mat()
        self.seams_rgb[~self.cumm_mask] = (1, 0, 0)

    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        super().seams_removal_horizontal(num_remove)
        
    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        super().seams_removal_vertical(num_remove)
    
    def backtrack_seam(self):
        """ Backtracks a seam for Seam Carving as taught in lecture
        """
        min_seam_idx = np.argmin(self.M[-1])
        gs_roll_left = np.roll(self.resized_gs, -1, axis=1)
        gs_roll_right = np.roll(self.resized_gs, 1, axis=1)
        gs_roll_down = np.roll(self.resized_gs, 1, axis=0)
        m_roll_left = np.roll(self.M, -1, axis=1)
        m_roll_right = np.roll(self.M, 1, axis=1)
        c_v = np.abs(gs_roll_left - gs_roll_right)
        c_l = np.abs(gs_roll_left - gs_roll_right) + np.abs(gs_roll_down - gs_roll_right)
        c_r = np.abs(gs_roll_left - gs_roll_right) + np.abs(gs_roll_down - gs_roll_left)
        for row in range(self.h - 1, 0, -1):
            self.mask[row, min_seam_idx] = False
            self.cumm_mask[self.idx_map_v[row, min_seam_idx], self.idx_map_h[row, min_seam_idx]] = False
            
            left = self.E[row] + (m_roll_right[row - 1] + c_l[row])
            right = self.E[row] + (m_roll_left[row - 1] + c_r[row])
            # Handle first and last columns - remove c_v
            if min_seam_idx == 0:
                right = right - c_v[row]
            elif min_seam_idx == self.w - 1:
                left = left - c_v[row]
                
            if self.M[row, min_seam_idx] == left[min_seam_idx] and min_seam_idx > 0:
                min_seam_idx = min_seam_idx - 1
            elif self.M[row, min_seam_idx] == right[min_seam_idx] and min_seam_idx < self.w - 1:
                min_seam_idx = min_seam_idx + 1
        self.mask[0, min_seam_idx] = False
        self.cumm_mask[self.idx_map_v[0, min_seam_idx], self.idx_map_h[0, min_seam_idx]] = False

    def remove_seam(self):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """
        super().remove_seam()
    
    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seamn to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO: Implement SeamImage.seams_addition")
    
    def seams_addition_horizontal(self, num_add):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    def seams_addition_vertical(self, num_add):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_vertical")

    @staticmethod
    # @jit(nopython=True)
    def calc_bt_mat(M, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.
        
        Recommnded parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a rederence type. changing it here may affected outsde.
        """
        raise NotImplementedError("TODO: Implement SeamImage.calc_bt_mat")

def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    return np.multiply(orig_shape, scale_factors).astype(int)

def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    seam_img.reinit()
    orig_shape = shapes[0]
    new_shape = shapes[1]
    vertical_num_remove = orig_shape[1] - new_shape[1]
    horizontal_num_remove = orig_shape[0] - new_shape[0]
    if vertical_num_remove > 0:
        seam_img.seams_removal_vertical(vertical_num_remove)
    if horizontal_num_remove > 0:
        seam_img.seams_removal_horizontal(horizontal_num_remove)
    
    return seam_img.resized_rgb
    
def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)
    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org
    scaled_x_grid = [get_scaled_param(x,in_width,out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y,in_height,out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid,dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid,dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:,x1s] * dx + (1 - dx) * image[y1s][:,x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:,x1s] * dx + (1 - dx) * image[y2s][:,x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image
