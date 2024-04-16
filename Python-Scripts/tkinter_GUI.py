from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import numpy as np
from skimage import feature
import pickle

# Function to load the model
def load_level_model(level, model_number):
    model_path = f"/home/amitabh/PycharmProjects/PLANTS-DISEASE-IDENTIFICATIN-CASCADED-MODEL/Level_{level}_model/level_{level}_model_no_{model_number}.pkl"
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

# Function to extract features from the image
def extract_features(image, n_points, radius, method):
    resized_image = image.resize((100, 100))
    gray_image = resized_image.convert("L")
    lbp_image = feature.local_binary_pattern(np.array(gray_image), n_points, radius, method)
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= lbp_hist.sum()  # Normalize

    X_predict = np.array([lbp_hist])

    return X_predict

# Function to perform the classification and display the result
def algorithm(image_path, n_points, radius, method):
    image = Image.open(image_path)
    features = extract_features(image, n_points, radius, method)
    model = load_level_model(level=0, model_number=0)
    prediction0 = model.predict(features)
    class_name = prediction0[0]
    if prediction0[0] == "Apple":
        model = load_level_model(level=1, model_number=0)
        prediction1 = model.predict(features)
        if prediction1[0] == "Apple Healthy":
            result_text = f"This leaf belongs to healthy {class_name} plant"
        else:
            result_text = f"This leaf belongs to {class_name} plant and has disease {prediction1[0]}"
        messagebox.showinfo("Classification Result", result_text)
    elif (prediction0[0] == "Bell pepper"):
        model = load_level_model(level=1, model_number=1)
        prediction1 = model.predict(features)
        if (prediction1[0] == "Bell pepper Healthy"):
            result_text = f"This leaf belongs to healthy {class_name} plant"
        else:
            result_text = f"This leaf belongs to {class_name} plant and has disease {prediction1[0]}"
        messagebox.showinfo("Classification Result", result_text)
    elif (prediction0[0] == "Cherry"):
        model = load_level_model(level=1, model_number=2)
        prediction1 = model.predict(features)
        if (prediction1[0] == "Cherry Healthy"):
            result_text = f"This leaf belongs to healthy {class_name} plant"
        else:
            result_text = f"This leaf belongs to {class_name} plant and has disease {prediction1[0]}"
        messagebox.showinfo("Classification Result", result_text)

    elif (prediction0[0] == "Corn"):
        model = load_level_model(level=1, model_number=3)
        prediction1 = model.predict(features)
        if (prediction1[0] == "Corn Healthy"):
            result_text = f"This leaf belongs to healthy {class_name} plant"
        else:
            result_text = f"This leaf belongs to {class_name} plant and has disease {prediction1[0]}"
        messagebox.showinfo("Classification Result", result_text)
    elif (prediction0[0] == "Grape"):
        model = load_level_model(level=1, model_number=4)
        prediction1 = model.predict(features)
        if (prediction1[0] == "Grape Healthy"):
            result_text = f"This leaf belongs to healthy {class_name} plant"
        else:
            result_text = f"This leaf belongs to {class_name} plant and has disease {prediction1[0]}"
        messagebox.showinfo("Classification Result", result_text)
    elif (prediction0[0] == "Peach"):
        model = load_level_model(level=1, model_number=5)
        prediction1 = model.predict(features)
        if (prediction1[0] == "Peach Healthy"):
            result_text = f"This leaf belongs to healthy {class_name} plant"
        else:
            result_text = f"This leaf belongs to {class_name} plant and has disease {prediction1[0]}"
        messagebox.showinfo("Classification Result", result_text)
    elif (prediction0[0] == "Potato"):
        model = load_level_model(level=1, model_number=6)
        prediction1 = model.predict(features)
        if (prediction1[0] == "Potato Healthy"):
            result_text = f"This leaf belongs to healthy {class_name} plant"
        else:
            result_text = f"This leaf belongs to {class_name} plant and has disease {prediction1[0]}"
        messagebox.showinfo("Classification Result", result_text)
    elif (prediction0[0] == "Strawberry"):
        model = load_level_model(level=1, model_number=7)
        prediction1 = model.predict(features)
        if (prediction1[0] == "Strawberry Healthy"):
            result_text = f"This leaf belongs to healthy {class_name} plant"
        else:
            result_text = f"This leaf belongs to {class_name} plant and has disease {prediction1[0]}"
        messagebox.showinfo("Classification Result", result_text)
    else:
        messagebox.showwarning("NO Matching plant found ")

# Function to open file dialog and select image
def select_image():
    filename = filedialog.askopenfilename(
        initialdir="/home/amitabh/PycharmProjects/PLANTS-DISEASE-IDENTIFICATIN-CASCADED-MODEL/Test-Images",
        title="Select Image",
        filetypes=(("All files", "*.*"),)
    )
    if filename:
        img = Image.open(filename)
        img = img.resize((100, 100))
        img = ImageTk.PhotoImage(img)
        lbl_show_pic.image = img
        lbl_show_pic['image'] = img
        entry_pic_path.delete(0, END)
        entry_pic_path.insert(0, filename)

# Function to classify the selected image
def classify_image():
    image_path = entry_pic_path.get()
    if not image_path or not Image.open(image_path):
        messagebox.showerror("Error", "Please select a valid image file.")
        return

    algorithm(image_path, n_points, radius, method)

n_points = 24
radius = 3
method = 'uniform'
# GUI setup
root = Tk()
root.geometry("800x600")
root.title("Plant Disease Identification")

frame = Frame(root)
frame.pack()



lbl_pic_path = Label(frame, text='Image Path:', font=('verdana', 16))
lbl_show_pic = Label(frame)
entry_pic_path = Entry(frame, font=('verdana', 16))
btn_browse = Button(frame, text='Select Image', bg='grey', fg='#ffffff', font=('verdana', 16), command=select_image)
btn_predict = Button(frame, text='Classify', bg='green', fg='#ffffff', font=('verdana', 16), command=classify_image)

lbl_pic_path.grid(row=0, column=0, pady=(10, 0))
entry_pic_path.grid(row=0, column=1, padx=(0, 20), pady=(10, 0))
lbl_show_pic.grid(row=1, column=0, columnspan="2")
btn_browse.grid(row=2, column=0, columnspan="2", padx=10, pady=10)
btn_predict.grid(row=3, column=0, columnspan="2", pady=10)

root.mainloop()
