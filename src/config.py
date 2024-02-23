INPUT_PATH = './input'

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_RESIZE = 96  # in pixels

# train-test split
TEST_SPLIT = 0.2

# show dataset keypoint plot
SHOW_DATASET_PLOT = True


keypoints = (
    'right_eye_center',
    'left_eye_center',
    'right_eye_inner_corner',
    'right_eye_outer_corner',
    'left_eye_inner_corner',
    'left_eye_outer_corner',
    'right_eyebrow_inner_end',
    'right_eyebrow_outer_end',
    'left_eyebrow_inner_end',
    'left_eyebrow_outer_end',
    'nose_tip',
    'mouth_right_corner',
    'mouth_left_corner',
    'mouth_center_top_lip',
    'mouth_center_bottom_lip'
)

