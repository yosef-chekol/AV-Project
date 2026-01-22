#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError


class SmartDriver(object):
    def __init__(self):
        rospy.init_node('smart_driver')

        self.bridge = CvBridge()

        # ✅ Camera topic (תתאים אם צריך)
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.image_callback, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.move = Twist()

        # ================== CONFIG ==================
        self.ROI_Y_START_RATIO = 0.55

        # Lane thresholds
        self.MIN_PIXELS_LINE = 250     # מינימום פיקסלים כדי להחשיב קו "נראה"
        self.LANE_HALF_WIDTH_FALLBACK = 130  # הסטה אם רואים רק קו אחד

        # Control
        self.KP = 0.0025
        self.MAX_SPEED = 0.18
        self.MIN_SPEED = 0.06
        self.SPEED_ERROR_GAIN = 0.002
        self.SMOOTH_ALPHA = 0.7  # smoothing error

        # RED sign detect (anti-stuck)
        self.RED_PIXELS_DETECT = 2600      # סף גילוי
        self.RED_PIXELS_CLEAR  = 900       # סף "נעלם"
        self.RED_CONFIRM_FRAMES = 4        # חייב להופיע כמה פריימים ברצף
        self.RED_COOLDOWN = 5.0            # לא מאפשר STOP חדש מיד אחרי

        # Stop behavior
        self.STOP_DURATION = 3.0

        # Overtake/correction behavior (offset-based, stays between lanes)
        self.OVERTAKE_OFFSET = 80          # פיקסלים (תשחק 60-110)
        self.OFFSET_DECAY_PER_FRAME = 8    # כמה מחזירים לכיוון 0 בכל פריים
        self.MIN_OVERTAKE_TIME = 1.0       # לפחות זמן קצר של "עקיפה" לפני שמתחילים להחזיר לאפס

        # ================== STATE ==================
        self.state = "FOLLOW"              # FOLLOW / STOP / OVERTAKE
        self.state_t0 = 0.0

        self.prev_error = 0.0

        # Red latch to prevent repeated stopping
        self.red_latched = False
        self.last_stop_time = -999.0
        self.red_seen_count = 0

        # Offset to "overtake" while still lane-following
        self.target_offset = 0             # + שמאלה, - ימינה
        self.overtake_dir = 1              # 1 left, -1 right

        # Optional: downsample processing like official code (reduce latency)
        self.frame_counter = 0
        self.PROCESS_EVERY_N_FRAMES = 1    # אם יש לאג, שים 2 או 3

        print("[SYSTEM] Smart Driver ready (STOP 3s + offset-based overtake)")

    def now(self):
        # ✅ ROS time (simulation friendly)
        return rospy.get_time()

    def image_callback(self, msg):
        self.frame_counter += 1
        if self.PROCESS_EVERY_N_FRAMES > 1:
            if (self.frame_counter % self.PROCESS_EVERY_N_FRAMES) != 0:
                return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            return

        h, w, _ = cv_image.shape
        screen_center = w / 2.0
        now = self.now()

        # ================= STOP STATE =================
        if self.state == "STOP":
            if (now - self.state_t0) < self.STOP_DURATION:
                self.publish_cmd(0.0, 0.0)
                return
            else:
                # Stop finished -> start "overtake correction" (still lane following)
                self.state = "OVERTAKE"
                self.state_t0 = now
                print("[STATE] OVERTAKE (offset lane-following)")

        # ================= ROI & LANE DETECTION =================
        roi = cv_image[int(h * self.ROI_Y_START_RATIO):h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_yellow = cv2.inRange(hsv, np.array([15, 60, 60]), np.array([40, 255, 255]))
        mask_white  = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([179, 30, 255]))

        # Count pixels for reliability
        yellow_count = int(np.count_nonzero(mask_yellow))
        white_count  = int(np.count_nonzero(mask_white))

        # lane center calculation (robust)
        lane_center = screen_center  # fallback

        if yellow_count > self.MIN_PIXELS_LINE:
            ys_y, xs_y = np.where(mask_yellow > 0)
            cx_y = float(np.mean(xs_y))
        else:
            cx_y = None

        if white_count > self.MIN_PIXELS_LINE:
            ys_w, xs_w = np.where(mask_white > 0)
            cx_w = float(np.mean(xs_w))
        else:
            cx_w = None

        if (cx_y is not None) and (cx_w is not None):
            lane_center = (cx_y + cx_w) / 2.0
        elif (cx_y is not None) and (cx_w is None):
            lane_center = cx_y + self.LANE_HALF_WIDTH_FALLBACK
        elif (cx_w is not None) and (cx_y is None):
            lane_center = cx_w - self.LANE_HALF_WIDTH_FALLBACK
        else:
            # Lost: slow search
            self.publish_cmd(0.05, 0.25)
            return

        # ================= RED SIGN DETECTION (with latch) =================
        # detect red on FULL image (not only roi) to be robust
        hsv_full = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask_red1 = cv2.inRange(hsv_full, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv_full, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        red_pixels = int(cv2.countNonZero(mask_red))

        # clear latch only after red really disappears
        if self.red_latched and red_pixels < self.RED_PIXELS_CLEAR:
            self.red_latched = False
            self.red_seen_count = 0
            # אחרי שהתמרור נעלם, נחזיר offset בהדרגה בתוך OVERTAKE/FOLLOW
            # (לא מאפסים מיד כדי לא "לקפוץ")
            print("[RED] Cleared (unlatched)")

        # confirm detection across several frames to avoid flicker
        if red_pixels > self.RED_PIXELS_DETECT:
            self.red_seen_count += 1
        else:
            self.red_seen_count = max(0, self.red_seen_count - 1)

        red_confirmed = (self.red_seen_count >= self.RED_CONFIRM_FRAMES)

        # Trigger STOP only if:
        # - we are FOLLOW
        # - red is confirmed
        # - not latched
        # - cooldown passed
        if (self.state == "FOLLOW") and red_confirmed and (not self.red_latched) and ((now - self.last_stop_time) > self.RED_COOLDOWN):
            # decide overtake direction based on sign position vs lane_center
            # compute sign cx
            M_r = cv2.moments(mask_red)
            if M_r['m00'] > 0:
                sign_cx = float(M_r['m10'] / M_r['m00'])

                # אם התמרור משמאל למרכז הנתיב => נזוז ימינה (offset שלילי)
                if sign_cx < lane_center:
                    self.overtake_dir = -1
                    decision = "RIGHT"
                else:
                    self.overtake_dir = 1
                    decision = "LEFT"

                # set offset target
                self.target_offset = self.OVERTAKE_OFFSET * self.overtake_dir

                self.state = "STOP"
                self.state_t0 = now
                self.last_stop_time = now
                self.red_latched = True  # prevent repeated stop while still seeing it
                print("[STOP] Red detected -> STOP 3s | decision={} | offset={}".format(decision, self.target_offset))
                self.publish_cmd(0.0, 0.0)
                return

        # ================= OFFSET MANAGEMENT =================
        # In OVERTAKE state we keep offset for minimum time, then decay it toward 0.
        if self.state == "OVERTAKE":
            if (now - self.state_t0) > self.MIN_OVERTAKE_TIME:
                # start returning offset back to 0 gradually
                if self.target_offset > 0:
                    self.target_offset = max(0, self.target_offset - self.OFFSET_DECAY_PER_FRAME)
                elif self.target_offset < 0:
                    self.target_offset = min(0, self.target_offset + self.OFFSET_DECAY_PER_FRAME)

                # when offset returned, go back to FOLLOW
                if self.target_offset == 0:
                    self.state = "FOLLOW"
                    print("[STATE] FOLLOW (offset finished)")

        # ================= LANE FOLLOW CONTROL (with offset) =================
        # key change: target is lane_center + offset
        target_center = lane_center + float(self.target_offset)

        raw_error = target_center - screen_center
        error = self.SMOOTH_ALPHA * self.prev_error + (1.0 - self.SMOOTH_ALPHA) * raw_error
        self.prev_error = error

        angular_z = -error * self.KP
        linear_x = max(self.MIN_SPEED, self.MAX_SPEED - abs(error) * self.SPEED_ERROR_GAIN)

        self.publish_cmd(linear_x, angular_z)

        # ================= DEBUG VIEW =================
        try:
            dbg = roi.copy()
            cv2.circle(dbg, (int(lane_center), 40), 6, (0, 255, 0), -1)        # lane center
            cv2.circle(dbg, (int(target_center), 40), 6, (255, 0, 255), -1)    # target with offset
            cv2.imshow("Smart View", dbg)
            cv2.waitKey(1)
        except Exception:
            pass

    def publish_cmd(self, lx, az):
        self.move.linear.x = float(lx)
        self.move.angular.z = float(az)
        self.cmd_vel_pub.publish(self.move)


if __name__ == "__main__":
    try:
        SmartDriver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
