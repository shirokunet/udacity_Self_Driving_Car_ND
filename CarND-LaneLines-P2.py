import numpy as np
import cv2
import pickle
from numba.decorators import jit
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip


class lane_detection:
    def __init__(self):
        # parameter
        self.image_size = [720, 1280]
        self.lpf_gain = 0.2

        self.start_flag = True
        self.l_limit_flag = False
        self.r_limit_flag = False

        self.left_fit = [0, 0, 350]
        self.right_fit = [0, 0, 1280 - 350]
        ploty = np.linspace(0, self.image_size[0] - 1, self.image_size[0])
        self.left_fitx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2]

        self.image_z1 = 0
        self.frame = 0

    @jit
    def caribration_image(self, img, image_dir='none', image_name='none'):
        dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        if image_dir != 'none':
            cv2.imwrite("calib_" + image_dir + image_name, undist)
        return undist

    @jit
    def transform_image(self, image, mode='none'):
        img_size = (image.shape[1], image.shape[0])
        # 1280, 720
        src = np.float32([[582, 460],
                          [309, 650],
                          [1009, 650],
                          [704, 460]])
        dst = np.float32([[(img_size[0] / 4), 0],
                          [(img_size[0] / 4), img_size[1]],
                          [(img_size[0] * 3 / 4), img_size[1]],
                          [(img_size[0] * 3 / 4), 0]])
        if mode == "r":
            M = cv2.getPerspectiveTransform(dst, src)
        else:
            M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, img_size)
        return warped

    @jit
    def multi_threshold(self, image, settings):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        combined_binary = np.zeros_like(gray)
        for s in settings:
            color_t = getattr(cv2, 'COLOR_RGB2{}'.format(s['cspace']))
            gray = cv2.cvtColor(image, color_t)[:, :, s['channel']]
            clahe = cv2.createCLAHE(s['clipLimit'], tileGridSize=(8, 8))
            norm_img = clahe.apply(gray)
            binary = np.zeros_like(norm_img)
            binary[(norm_img >= s['threshold']) & (norm_img <= 255)] = 1
            combined_binary[(combined_binary == 1) | (binary == 1)] = 1
        return combined_binary

    @jit
    def calc_half_bottom_histogram(self, image, white_noise=20):
        histogram_src = np.sum(image[image.shape[0] // 2:, :], axis=0)
        histogram_src = np.where(histogram_src < white_noise, 0, histogram_src)
        return histogram_src

    @jit
    def find_two_peaks(self, histogram, side_margin=250, midle_margin=150, value_th=1):
        midpoint = np.int(histogram.shape[0] // 2)
        xmax = np.int(histogram.shape[0])

        leftx_base = np.argmax(histogram[side_margin:midpoint - midle_margin]) + side_margin
        rightx_base = np.argmax(histogram[midpoint + midle_margin:xmax - side_margin]) + midpoint + midle_margin

        leftx_base_value = histogram[leftx_base]
        rightx_base_value = histogram[rightx_base]
        if leftx_base_value <= value_th and rightx_base_value <= value_th:
            leftx_base = 350
            rightx_base = 1280 - 350
        elif leftx_base_value <= value_th:
            leftx_base = xmax - rightx_base
        elif rightx_base_value <= value_th:
            rightx_base = xmax - leftx_base

        return leftx_base, rightx_base

    @jit
    def calc_nonzero_points(self, image):
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        return nonzeroy, nonzerox

    @jit
    def calc_dstack(self, image):
        return np.dstack((image, image, image)) * 255

    @jit
    def moving_window(self, binary_warped, nonzeroy, nonzerox,
                      leftx_base, rightx_base,
                      left_fit, right_fit, l_limit_flag, r_limit_flag,
                      nwindows=9, margin=100, minpix=100,):
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] // nwindows)

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        out_img = binary_warped.copy()
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = int(binary_warped.shape[0] - (window + 1) * window_height)
            win_y_high = int(binary_warped.shape[0] - window * window_height)

            if (left_fit[0] == 0 and left_fit[1] == 0)or l_limit_flag:
                win_xleft_low = int(leftx_current - margin)
                win_xleft_high = int(leftx_current + margin)
            else:
                win_xleft_base = left_fit[0] * win_y_high * win_y_high + left_fit[1] * win_y_high + left_fit[2]
                win_xleft_low = int(win_xleft_base - margin)
                win_xleft_high = int(win_xleft_base + margin)

            if (right_fit[0] == 0 and right_fit[1] == 0) or r_limit_flag:
                win_xright_low = int(rightx_current - margin)
                win_xright_high = int(rightx_current + margin)
            else:
                win_xright_base = right_fit[0] * win_y_high * win_y_high + right_fit[1] * win_y_high + right_fit[2]
                win_xright_low = int(win_xright_base - margin)
                win_xright_high = int(win_xright_base + margin)

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 5)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 5)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            # 探索領域内の有効要素を後のフィッティングのために貯めこんでいく
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            # 窓内で発見した左右の車線位置
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        # 配列の連結
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        return left_lane_inds, right_lane_inds, out_img

    # @jit
    def calc_fitting(self, nonzeroy, nonzerox, left_lane_inds, right_lane_inds,
                     left_fit_z1, right_fit_z1,
                     alim=0.01, blim=10, lane_width_min=400, lane_width_max=800):
        # left a,b,c [  9.00199449e-05  -2.72302356e-01   5.02215113e+02]
        # right a,b,c [  1.73323778e-04  -2.83396970e-01   1.12500427e+03]

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # use last abc to append half bottom area
        ploty = np.linspace(self.image_size[0] // 3 * 2, self.image_size[0] - 1, self.image_size[0] // 3 * 2)
        left_fitx_z1 = left_fit_z1[0] * ploty**2 + left_fit_z1[1] * ploty + left_fit_z1[2]
        right_fitx_z1 = right_fit_z1[0] * ploty**2 + right_fit_z1[1] * ploty + right_fit_z1[2]
        leftx = np.append(leftx, left_fitx_z1)
        lefty = np.append(lefty, ploty)
        rightx = np.append(rightx, right_fitx_z1)
        righty = np.append(righty, ploty)

        l_limit_flag = False
        r_limit_flag = False
        # Fit a second order polynomial to eachnonzerox
        # 二次関数でフィッティング
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except:
            left_fit = [0, 0, 350]
            right_fit = [0, 0, 1280 - 350]
            l_limit_flag = True
            r_limit_flag = True

        if left_fit[0] > alim:
            left_fit[0] = alim
            l_limit_flag = True
        elif left_fit[0] < -alim:
            left_fit[0] = -alim
            l_limit_flag = True
        if left_fit[1] > blim:
            left_fit[1] = blim
            l_limit_flag = True
        elif left_fit[1] < -blim:
            left_fit[1] = -blim
            l_limit_flag = True

        if right_fit[0] > alim:
            right_fit[0] = alim
            r_limit_flag = True
        elif right_fit[0] < -alim:
            right_fit[0] = -alim
            r_limit_flag = True
        if right_fit[1] > blim:
            right_fit[1] = blim
            r_limit_flag = True
        elif right_fit[1] < -blim:
            right_fit[1] = -blim
            r_limit_flag = True

        ymax = self.image_size[0]
        xmax = self.image_size[1]
        xmidle = xmax / 2
        left_lane_x = left_fit[0] * ymax * ymax + left_fit[1] * ymax + left_fit[2]
        right_lane_x = right_fit[0] * ymax * ymax + right_fit[1] * ymax + right_fit[2]
        if left_lane_x < 0 or left_lane_x > xmidle:
            l_limit_flag = True
        if right_lane_x < xmidle or right_lane_x > xmax:
            r_limit_flag = True

        return left_fit, right_fit, l_limit_flag, r_limit_flag

    @jit
    def calc_ane_inds_width_limit(self, nonzeroy, nonzerox, left_fit, right_fit, margin=50):
        left_lane_left_margin = (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)
        left_lane_right_margin = (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)
        left_lane_inds_width_limit = ((nonzerox > left_lane_left_margin) & (nonzerox < left_lane_right_margin))

        right_lane_left_margin = (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)
        right_lane_right_margin = (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin)
        right_lane_inds_width_limit = ((nonzerox > right_lane_left_margin) & (nonzerox < right_lane_right_margin))

        return left_lane_inds_width_limit, right_lane_inds_width_limit

    @jit
    def calc_fitpoints(self, image, left_fit, right_fit):
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        return left_fitx, right_fitx

    @jit
    def calc_window_img(self, image, left_fitx, right_fitx, margin=50, color=(0, 55, 0)):
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        window_img = image.copy()
        cv2.fillPoly(window_img, np.int_([left_line_pts]), color)
        cv2.fillPoly(window_img, np.int_([right_line_pts]), color)

        return window_img

    @jit
    def calc_road_img(self, image, left_fitx, right_fitx):
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))

        # Draw the lane onto the warped blank image
        window_img = np.zeros_like(image)
        cv2.fillPoly(window_img, np.int_([line_pts]), (0, 155, 0))
        return window_img

    @jit
    def combine_images(self, masked_image, line_image):
        if len(masked_image.shape) == 2:
            masked_image = np.dstack((masked_image, masked_image, masked_image))
        if len(line_image.shape) == 2:
            line_image = np.dstack((line_image, line_image, line_image))
        return cv2.addWeighted(masked_image, 1.0, line_image, 1.0, 1)

    @jit
    def input_overlay(self, image, overlay, org, border_thickness=2):
        if len(overlay.shape) == 2:
            overlay = np.dstack((overlay, overlay, overlay)) * 255

        # Add border to overlay
        overlay[:border_thickness, :, :] = 255
        overlay[-border_thickness:, :, :] = 255
        overlay[:, :border_thickness, :] = 255
        overlay[:, -border_thickness:, :] = 255

        # Place overlay onto image
        x_offset, y_offset = org
        image[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1]] = overlay

        return image

    def process_image(self, image):
        # caribration, transform
        image_caribrationed = self.caribration_image(image)
        image_transformed = self.transform_image(image_caribrationed)

        # filter
        settings = []
        settings.append({'cspace': 'LAB', 'channel': 2, 'clipLimit': 2.0, 'threshold': 145})
        settings.append({'cspace': 'HLS', 'channel': 1, 'clipLimit': 2.0, 'threshold': 205})
        settings.append({'cspace': 'HSV', 'channel': 2, 'clipLimit': 6.0, 'threshold': 215})
        image_threshold = self.multi_threshold(image_transformed, settings)

        # histogram
        bottom_histogram = self.calc_half_bottom_histogram(image_threshold)
        leftx_base, rightx_base = self.find_two_peaks(bottom_histogram)

        nonzeroy, nonzerox = self.calc_nonzero_points(image_threshold)
        image_threshold_dstack = self.calc_dstack(image_threshold)

        # find suitable area by using moving window
        left_lane_inds, right_lane_inds, image_window = self.moving_window(image_threshold_dstack,
                                                                           nonzeroy, nonzerox,
                                                                           leftx_base, rightx_base,
                                                                           self.left_fit, self.right_fit,
                                                                           self.l_limit_flag, self.r_limit_flag,
                                                                           nwindows=8, margin=80, minpix=50)

        # calc lane abc
        left_fit, right_fit, self.l_limit_flag, self.r_limit_flag = self.calc_fitting(nonzeroy, nonzerox,
                                                                                      left_lane_inds, right_lane_inds,
                                                                                      self.left_fit, self.right_fit)
        left_lane_inds_lim, right_fit_inds_lim = self.calc_ane_inds_width_limit(nonzeroy, nonzerox,
                                                                                self.left_fit, self.right_fit,
                                                                                margin=80)
        left_fit_lim, right_fit_lim, self.l_limit_flag, self.r_limit_flag = self.calc_fitting(nonzeroy, nonzerox,
                                                                                              left_lane_inds_lim, right_fit_inds_lim,
                                                                                              self.left_fit, self.right_fit)

        # lan lpf
        if self.start_flag:
            self.left_fit = left_fit
            self.right_fit = right_fit
            self.start_flag = False
        else:
            self.left_fit[0] = (left_fit_lim[0] * self.lpf_gain) + (self.left_fit[0] * (1.0 - self.lpf_gain))
            self.left_fit[1] = (left_fit_lim[1] * self.lpf_gain) + (self.left_fit[1] * (1.0 - self.lpf_gain))
            self.left_fit[2] = (left_fit_lim[2] * self.lpf_gain) + (self.left_fit[2] * (1.0 - self.lpf_gain))
            self.right_fit[0] = (right_fit_lim[0] * self.lpf_gain) + (self.right_fit[0] * (1.0 - self.lpf_gain))
            self.right_fit[1] = (right_fit_lim[1] * self.lpf_gain) + (self.right_fit[1] * (1.0 - self.lpf_gain))
            self.right_fit[2] = (right_fit_lim[2] * self.lpf_gain) + (self.right_fit[2] * (1.0 - self.lpf_gain))

        # calc lane array
        # pink
        left_fitx_raw, right_fitx_raw = self.calc_fitpoints(image_threshold_dstack, left_fit, right_fit)
        image_window_line = self.calc_window_img(image_window, left_fitx_raw, right_fitx_raw,
                                                 margin=5, color=(255, 0, 255))
        # blue
        left_fitx_lim, right_fitx_lim = self.calc_fitpoints(image_threshold_dstack, left_fit_lim, right_fit_lim)
        image_window_line = self.calc_window_img(image_window_line, left_fitx_lim, right_fitx_lim,
                                                 margin=5, color=(0, 255, 255))
        # yelow
        self.left_fitx, self.right_fitx = self.calc_fitpoints(image_threshold_dstack, self.left_fit, self.right_fit)
        image_window_line = self.calc_window_img(image_window_line, self.left_fitx, self.right_fitx,
                                                 margin=5, color=(255, 255, 0))
        # green
        image_transformed_line = self.calc_window_img(image_transformed, self.left_fitx, self.right_fitx,
                                                      margin=5, color=(0, 255, 0))

        # calc road area
        road_img = self.calc_road_img(image_threshold_dstack, self.left_fitx, self.right_fitx)

        road_transformed = self.transform_image(road_img, mode="r")
        image_caribrationed_road_area = self.combine_images(image_caribrationed, road_transformed)

        # overlay
        overhead_img = cv2.resize(image_transformed, None, fx=1 / 4.0, fy=1 / 4.0)
        self.input_overlay(image_caribrationed_road_area, overhead_img, (0, 0))
        overhead_img = cv2.resize(image_window, None, fx=1 / 4.0, fy=1 / 4.0)
        self.input_overlay(image_caribrationed_road_area, overhead_img, (image.shape[1] // 4, 0))
        overhead_img = cv2.resize(image_window_line, None, fx=1 / 4.0, fy=1 / 4.0)
        self.input_overlay(image_caribrationed_road_area, overhead_img, (image.shape[1] // 4 * 2, 0))
        overhead_img = cv2.resize(image_transformed_line, None, fx=1 / 4.0, fy=1 / 4.0)
        self.input_overlay(image_caribrationed_road_area, overhead_img, (image.shape[1] // 4 * 3, 0))

        # text
        text = str(self.frame)
        cv2.putText(image_caribrationed_road_area, text, org=(50, self.image_size[0] - 140),
                    fontScale=0.6, thickness=1, color=(255, 255, 0),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, lineType=cv2.LINE_AA)

        # store z1
        self.image_z1 = road_transformed
        self.frame += 1

        return image_caribrationed_road_area


if __name__ == '__main__':
    # ld = lane_detection()
    # white_output = 'test_videos_output/challenge_video.mp4'
    # clip1 = VideoFileClip("test_videos/challenge_video.mp4").subclip(0, 15)
    # white_clip = clip1.fl_image(lambda x: ld.process_image(x))
    # white_clip.write_videofile(white_output, audio=False)

    ld1 = lane_detection()
    white_output = 'test_videos_output/project_video.mp4'
    clip1 = VideoFileClip("test_videos/project_video.mp4").subclip(20, 50)
    white_clip = clip1.fl_image(lambda x: ld1.process_image(x))
    white_clip.write_videofile(white_output, audio=False)

    # ld2 = lane_detection()
    # white_output = 'test_videos_output/harder_challenge_video.mp4'
    # clip1 = VideoFileClip("test_videos/harder_challenge_video.mp4").subclip(0, 10)
    # white_clip = clip1.fl_image(lambda x: ld2.process_image(x))
    # white_clip.write_videofile(white_output, audio=False)
