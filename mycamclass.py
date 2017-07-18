'''
CLASSES
'''
class Cam:
    internal_count = [] #store all instances here!
    def __init__(self, name=None, camera_port=0, ramp_frames=30):
        global cam_count
        ### the laptop CAMERA
        # Camera 0 is the integrated web cam on my netbook
        self.camera_port = camera_port
        #Number of frames to throw away while the camera adjusts to light levels
        self.ramp_frames = ramp_frames
        # Now we can initialize the camera capture object with the cv2.VideoCapture class.
        # All it needs is the index to a camera port.
        self.camera = cv2.VideoCapture(camera_port)
        if name:
            if "%" in name:
                self.filename = gettext.gettext(name % cam_count)
                cam_count = cam_count + 1
            else:
                self.filename = name
        else:
            self.filename = "test_image%d.png" % cam_count
            cam_count = cam_count + 1
        self.get_image()
    #### CAPTURES #####
    # Captures a single image from the camera and returns it in PIL format
    def read_cam(self):
        #read is the easiest way to get a full image out of a VideoCapture object.
        retval, im = self.camera.read()
        return im
    def get_image(self):# Ramp the camera - these frames will be discarded and are only used to allow v4l2
        # to adjust light levels, if necessary
        for i in xrange(self.ramp_frames):
            temp = self.read_cam()
        print("Taking image...")
        # Take the actual image we want to keep
        camera_capture = self.read_cam()
        # A nice feature of the imwrite method is that it will automatically choose the
        # correct format based on the file extension you provide. Convenient!
        cv2.imwrite(self.filename, camera_capture)
        #cv2.imwrite('img_CV2_90.jpg', a, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        # You'll want to release the camera, otherwise you won't be able to create a new
        # capture object until your script exits
        del(self.camera)
        print("#### END CAPTURES for %s " % self.filename)
        #### END CAPTURES #####

## create cam once, but images name and folder and counter in get_image()?
#c = Cam('./conv3/test_convnn%s.png')
