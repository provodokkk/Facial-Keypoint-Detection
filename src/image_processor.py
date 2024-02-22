import os
import cv2
import csv
import shutil
import numpy as np

from typing import List, Tuple, Sequence
from tkinter import *
from tkinter.font import Font

from PIL import Image, ImageTk, ImageDraw

from config import IMG_RESIZE, keypoints


SAVE_PATH: str = r'..\processed_images'
VALIDATION_IMG_PATH: str = r'..\validation_images'
SCALER: int = 4


def get_cropped_faces(folder_path: str, img_name: str) -> List[np.ndarray]:
    img = cv2.imread(fr'{folder_path}\{img_name}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_faces = detect_faces(img)

    resized_faces_1_channel = get_processed_faces(img, detected_faces, three_channels=False)
    resized_faces_3_channels = get_processed_faces(img, detected_faces, three_channels=True)

    save_faces(img_name, resized_faces_3_channels)

    # Flatten the images to lists
    return [face.flatten().tolist() for face in resized_faces_1_channel]


def detect_faces(img: np.ndarray) -> Sequence[Sequence[int]]:
    # Load the Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(r'..\haarcascade_frontalface_alt2.xml')

    # Return detected faces
    return face_cascade.detectMultiScale(img, 1.1, 4)


def get_processed_faces(img: np.ndarray, detected_faces, three_channels: bool = False) -> List[np.ndarray]:
    if three_channels:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cropped_faces = [img[y:y + h, x:x + w] for (x, y, w, h) in detected_faces]
    resized_faces = [cv2.resize(face, (96, 96)) for face in cropped_faces]

    return resized_faces


def save_faces(img_name: str, faces) -> None:
    file_name = get_file_name_without_extension(img_name)
    os.makedirs(SAVE_PATH, exist_ok=True)

    for i, face in enumerate(faces):
        new_img_path = fr'{SAVE_PATH}\{file_name}_{i}.jpg'
        cv2.imwrite(new_img_path, face)


def get_file_name_without_extension(full_file_name: str) -> str:
    return os.path.splitext(full_file_name)[0]


def reset_folder(folder_path: str = SAVE_PATH) -> None:
    """
    Deletes the specified folder and all its contents, then recreates an empty folder with the same path.

    Args:
        folder_path: The path to the folder to reset.
    """
    try:
        shutil.rmtree(folder_path)
        print(f"Folder {folder_path} has been deleted.")
    except FileNotFoundError:
        print(f"Folder {folder_path} does not exist, creating it.")
    except Exception as e:
        print(f"Failed to delete {folder_path}. Reason: {e}")
        return

    try:
        os.makedirs(folder_path)
        print(f"Folder {folder_path} has been recreated.")
    except Exception as e:
        print(f"Failed to recreate {folder_path}. Reason: {e}")


def convert_coordinates(x: int, y: int) -> Tuple[float, float]:
    x = float(x) / SCALER
    y = float(y) / SCALER
    return (x if x < 96.0 else 96.0), (y if y < 96.0 else 96.0)


def append_data_to_csv(csv_path: str, coords: list[float], img: np.ndarray) -> None:
    """
    Appends a list of coordinates and image to the end of a CSV file.

    Args:
        csv_path: The file path to the CSV file.
        coords: A list of coordinates to be appended.
        img:
    """
    img_str = ' '.join(map(str, img))
    space_separated_img = img_str.replace(',', ' ')
    row_data = coords + [space_separated_img]

    print(coords)

    with open(csv_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row_data)

class ImageAnnotator:
    def __init__(self, image_file):
        self.win = Tk()
        self.canvas = Canvas(self.win, width=IMG_RESIZE * SCALER, height=IMG_RESIZE * SCALER)
        self.canvas.pack()

        # Load and resize the image for display
        img_path = os.path.join(SAVE_PATH, image_file)
        self.img = Image.open(img_path)
        self.img_resized = self.img.resize((IMG_RESIZE * SCALER, IMG_RESIZE * SCALER), Image.NEAREST)

        self.img_tk = ImageTk.PhotoImage(self.img_resized)
        self.canvas.create_image(0, 0, anchor=NW, image=self.img_tk)

        # A label for instructions
        self.kp_index = 0
        font = Font(family='San Francisco', size=16, weight='bold')
        self.status_label = Label(self.win, text=keypoints[self.kp_index], font=font, fg='#D5331E')
        self.status_label.pack()

        self.image_file = image_file
        self.keypoint_coordinates = []
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def on_canvas_click(self, event):
        """Handle mouse click event on canvas."""

        # Draw a red point on the image at the clicked location
        img_x, img_y = event.x, event.y
        self.draw_point(img_x, img_y)

        # Update the canvas to reflect the drawing
        self.update_canvas()

        # Save the clicked point
        self.save_point(img_x, img_y)

        # Update label and check if all keypoints have been annotated
        self.kp_index += 1
        if self.kp_index < len(keypoints):
            self.status_label.config(text=keypoints[self.kp_index])
        else:
            # Save the image with keypoints when all points are annotated
            self.save_image()
            self.win.destroy()

    def draw_point(self, x, y, radius=5, color='red'):
        draw = ImageDraw.Draw(self.img_resized)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    def save_point(self, x, y):
        x, y = convert_coordinates(x, y)
        self.keypoint_coordinates.extend([x, y])

    def update_canvas(self):
        self.img_tk = ImageTk.PhotoImage(self.img_resized)
        self.canvas.create_image(0, 0, anchor=NW, image=self.img_tk)

    def get_image_data(self):
        return self.keypoint_coordinates, self.img

    def save_image(self):
        img_name = get_file_name_without_extension(self.image_file)
        annotated_img_path = os.path.join(VALIDATION_IMG_PATH, f'{img_name}_with_keypoints.jpg')
        self.img_resized.save(annotated_img_path)

    def run(self):
        self.win.mainloop()


def get_all_images_in_folder():
    return [f for f in os.listdir(SAVE_PATH) if f.endswith('.jpg')]


if __name__ == '__main__':
    reset_folder(SAVE_PATH)

    faces = get_cropped_faces('..', 'img.jpg')
    image_files = get_all_images_in_folder()

    for image_file, face in zip(image_files, faces):
        annotator = ImageAnnotator(image_file)
        annotator.run()

        keypoint_coordinates, img = annotator.get_image_data()
        append_data_to_csv(r'..\input\empty.csv', keypoint_coordinates, face)
