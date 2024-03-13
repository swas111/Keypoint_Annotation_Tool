import os
import cv2
import tkinter as tk
from tkinter import Canvas, Button, Label, Toplevel
import math
from ultralytics import YOLO

model = YOLO(r"weights/last.pt")

colors = [(0,0,0),(255,255,255),(128,128,128),
          (139,69,19),(244,164,96),(0,100,0),
          (0,0,139),(0,128,0), (0,0,255),
          (0,255,0) ,(30,144,255),  (139,0,0),
          (128,0,128),(255,0,0),(255,20,147),
          (255,127,80) , (255,192,203) , (173,255,47),
          (135,206,250),(0,128,128),(255,215,0),
          (216,191,216)

          ]





parts = [
    "left eye", "right eye", "nose", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left hip",
    "right hip", "left knee", "right knee", "left ankle", "right ankle",
    "left finger", "right finger", "back", "left toe", "right toe"
]

class KeypointEditor:
    def __init__(self, root,folder_path):
        # initial_points = initial_points[0]
        self.root = root
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_size = 256
        self.image_ac_size = None
        # self.image_path = image_path
        self.initial_points = []
        self.final_points = []  # Convert to lists
        self.selected_point = None
        self.current_image_index = 0
        self.labels_folder = ""
        self.save_path = ""
        self.scale_factor = 2.5  # Scale factor for both image and points
        # image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])

        # self.label = Label(root, text="File Name: {}".format(os.path.basename(image_path)))
        # self.label.pack()
        self.label = Label( root ,text="")
        self.label.pack()
        self.message_label = Label(root, text="")
        self.messagesave_label = Label(root , text="j")
        self.size_label = Label(root, text="")
        self.size_label.pack()

        self.message_label.pack()
        self.messagesave_label.pack()
        self.canvas = Canvas(root)
        self.canvas.pack()
        self.connections = [
            (0, 1), (1, 2), (0, 2),  # Left eye, right eye, nose
            (0, 3), (1, 4),                # Left ear, right ear
            (5, 7), (7 ,9), (9,17)  ,         # Left Arm
            (6, 8), (8, 10), (10 ,18) ,          # Right Arm
            (11, 13), (13, 15), (15 , 20) ,      # left Leg 
            (12, 14), (14, 16), (16 , 21) ,      # Left knee, right knee
            (11 , 19) , (12 , 19) , (5 , 19) , (6, 19)
        ]
        self.load_image()
        self.draw_connections(self.connections)
        self.draw_keypoints()
        self.check_file()

        self.save_button = Button(root, text="Save", command=self.save_coordinates)
        self.save_button.pack()
        self.next_button = Button(root, text="Next", command=self.next_image)
        self.prev_button = Button(root, text="Previous", command=self.prev_image)
        self.delete_button = Button(root, text="Delete", command=self.delete_current_image)

        self.pack_buttons()

        self.canvas.bind("<Button-1>", self.select_point)
        self.canvas.bind("<B1-Motion>", self.move_selected_point)
    def pack_buttons(self):
        self.next_button.pack(side=tk.RIGHT)
        self.prev_button.pack(side=tk.LEFT)
        self.delete_button.pack()

    def load_image(self):
        image_path1 = os.path.join(self.folder_path, self.image_files[self.current_image_index])
        self.label.config(text="File Name: {}".format(os.path.basename(image_path1)))
        image = cv2.resize(cv2.imread(image_path1), (256, 256))
        results = model(image)
        
        # if results and results[0].keypoints is not None:
        self.initial_points = results[0].keypoints.xy.tolist()
        initial_points = self.initial_points[0]
        self.final_points = [list(point) for point in initial_points]
        self.image = cv2.imread(image_path1)
        image_size = self.image.shape
        self.size_label.config(text=f"Image_ac_size {image_size}")
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = cv2.resize(self.image, None, fx=self.scale_factor, fy=self.scale_factor)  # Scale up the image
        self.photo = tk.PhotoImage(data=cv2.imencode('.png', self.image)[1].tobytes())
        self.canvas.config(width=self.image.shape[1], height=self.image.shape[0])  # Set canvas size to match image size
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def draw_keypoints(self):
        for i, (x, y) in enumerate(self.final_points):
            x_scaled, y_scaled = x * self.scale_factor, y * self.scale_factor  # Scale up the points
            color = colors[i % len(colors)]
            self.canvas.create_oval(x_scaled - 5, y_scaled - 5, x_scaled + 5, y_scaled + 5,
                                    fill="#%02x%02x%02x" % color, outline="#%02x%02x%02x" % color)
            

    def draw_connections(self, connections):
        for connection in connections:
            i, j = connection
            x1, y1 = self.final_points[i]
            x2, y2 = self.final_points[j]
            x1_scaled, y1_scaled = x1 * self.scale_factor, y1 * self.scale_factor
            x2_scaled, y2_scaled = x2 * self.scale_factor, y2 * self.scale_factor
            self.canvas.create_line(x1_scaled, y1_scaled, x2_scaled, y2_scaled, fill="white")

    def select_point(self, event):
        x, y = event.x // self.scale_factor, event.y // self.scale_factor  # Scale down the mouse click coordinates
        for i, (px, py) in enumerate(self.final_points):
            distance = math.sqrt((px - x)**2 + (py - y)**2)
            if distance < 5:
                self.selected_point = i
                break
    def move_selected_point(self, event):
        if self.selected_point is not None:
            x, y = event.x // self.scale_factor, event.y // self.scale_factor  # Scale down the mouse move coordinates
            self.final_points[self.selected_point] = [int(x), int(y)]  # Convert to integers and update as list
            self.redraw_keypoints()
            self.redraw_connections()


    def redraw_keypoints(self):
        self.canvas.delete("all")
        self.load_image()
        self.draw_keypoints()

    def redraw_connections(self):
        self.canvas.delete("connections")
        self.draw_connections(self.connections)
    

    def save_coordinates(self):
        # Ensure all points in final_points are integers
        self.final_points = [[int(px), int(py)] for px, py in self.final_points]

        # Normalize the points to the original image size
        normalized_points = [(x / self.image_size, y / self.image_size) for x, y in self.final_points]

        # Create a formatted string for saving
        save_string = "0 0.5 0.5 1 1 " + " ".join([f"{x} {y}" for x, y in normalized_points])

        # Create a 'labels' folder if it doesn't exist
        image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])

        labels_folder = os.path.join(os.path.dirname(image_path), "labels")
        os.makedirs(labels_folder, exist_ok=True)

        # Save to a .txt file inside the 'labels' folder
        save_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        save_path = os.path.join(labels_folder, save_filename)
        
          # Update the message label
        with open(save_path, "w") as file:
            file.write(save_string)

        # Update the message label
        self.message_label.config(text=f"Saved coordinates to {save_path}")
    def next_image(self):
        # print("Next button pressed")
        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
        self.load_image()
        self.draw_connections(self.connections)
        self.draw_keypoints()
        self.check_file()
        self.message_label.config(text="")
    def check_file(self):
        image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])

        labels_folder = os.path.join(os.path.dirname(image_path), "labels")
        os.makedirs(labels_folder, exist_ok=True)
        save_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        save_path = os.path.join(labels_folder, save_filename)
        txt_files = os.listdir(labels_folder)
        # jpg_files = os.listdir(self.folder_path)
        jpg_file = self.image_files[self.current_image_index]
        if jpg_file[:-4] not in [f.split('.')[0] for f in txt_files if f.endswith('.txt')]:
            self.messagesave_label.config(text = f"coordinates not there {jpg_file}")
        else :
            self.messagesave_label.config(text = f"coordinates saved {jpg_file}" )
    def prev_image(self):
        self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
        if self.current_image_index < 0:
            self.current_image_index = len(self.image_files) - 1
        self.load_image()
        self.draw_connections(self.connections)
        self.draw_keypoints()
        self.message_label.config(text="")
        self.check_file()

    def delete_current_image(self):
        file_path_d = os.path.join(self.folder_path, self.image_files[self.current_image_index])
        if self.current_image_index < len(self.image_files):
            os.remove(os.path.join(self.folder_path, self.image_files[self.current_image_index]))
            self.image_files.pop(self.current_image_index)

        if self.current_image_index >= len(self.image_files):
            self.current_image_index = 0

        self.load_image()
        self.draw_connections(self.connections)
        self.draw_keypoints()
        self.message_label.config(text=f" {file_path_d} file deleted")

    # @property
    # def image_files(self):
    #     return [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def display_parts_colors():
    part_colors_window = Toplevel()
    part_colors_window.title("Parts and Colors")

    for i in range(len(parts)):
        color = colors[i % len(colors)]
        part_label = Label(part_colors_window, text=parts[i], fg="#%02x%02x%02x" % color)
        part_label.grid(row=i, column=0, sticky="w")

        color_label = Label(part_colors_window, text="#%02x%02x%02x" % color)
        color_label.grid(row=i, column=1, sticky="e")

def main():
    folder_path = "trial2"
    # image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # for image_file in image_files:
    #     image_path = os.path.join(folder_path, image_file)
    #     image = cv2.resize(cv2.imread(image_path), (256, 256))
    #     results = model(image)
        
    #     # if results and results[0].keypoints is not None:
    #     initial_points = results[0].keypoints.xy.tolist()
        # else:
            # print("Error: results is empty or keypoints is not defined
    root = tk.Tk()
    root.title("Keypoint Editor")
    editor = KeypointEditor(root,folder_path)
    parts_button = Button(root, text="Display Parts and Colors", command=display_parts_colors)
        # delete_btn = Button(root, text="delete all" , command=redraw_keypoints )
    parts_button.pack()
    root.mainloop()

if __name__ == "__main__":
    main()
