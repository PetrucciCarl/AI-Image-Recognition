import torch
import torchvision
import tkinter
import PIL
import io
import os
import sys
from tkinter import ttk, filedialog

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

#if possible move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformation = torchvision.transforms.Compose([
    #squash and stretch images into a fixed size (224 * 224 pixels)
    torchvision.transforms.Resize((224, 224)),
    #convert image pixel data into tensor data in the format of [Color, Height, Width]
    torchvision.transforms.ToTensor(),
    #adjust the data to account for differences in things such as brightness and contrast to avoid confusion during training, these values were used in examples throughout the documentation for torchvision
    torchvision.transforms.Normalize(mean=[0.485, 0.456 , 0.406], std=[0.229, 0.224, 0.225])
])

#run this function instead of the draw function to retrain the model
def train(epochs):
    global model

    #get a pretrained model which can already recognise things like shapes, edges, lines, and objects etc...
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    #prevent the layers within the pre-trained model from being adjusted during training since they already work well
    #models are made up of layers, layers can either be 'linear layers' or 'activation layers'
    #linear layers have weights and biases which are adjusted during training, they are essentially linear equations (y = mx + c) where m and c are gradually adjusted for the model to try and match the expected output
    #activation layers re-shape the linear layers using different mathematical operations, they do not have weights or biases to adjust during training
    for parameter in model.parameters():
        parameter.requires_grad = False

    #replace the final layer so that the model so that the output can be a number corresponding to a folder
    #look at an output from the previous layer containing many different patterns, and squash it into 'number of options' possiple outputs
    model.fc = torch.nn.Linear(model.fc.in_features, len(os.listdir("images")))

    #load data from the images folder
    data = torchvision.datasets.ImageFolder("images", transform=transformation)

    #a data loader handles loading multiple images at a time (in a batch) and shuffling which images are grouped together and in what order (to prevent the model from learning patterns based on the order of images)
    training_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

    loss_function = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    model.to(device)

    #'epoch' refers to a full cycle of training through all of the data
    for epoch in range(epochs):

        print(f"starting epoch {epoch + 1}...")

        for images, labels in training_loader:
            #move images and labels to gpu if possible
            images = images.to(device)
            labels = labels.to(device)

            #make a prediction
            outputs = model(images)
            #calculate how wrong the prediction is
            loss = loss_function(outputs, labels)
            #reset the gradients
            optimizer.zero_grad()
            #compute the new gradients
            loss.backward()
            #update the weights
            optimizer.step()

        print(f"finished epoch {epoch + 1}!")

    #save the weights so that there is no need to retrain
    torch.save(model.state_dict(), "model.pth")

def load():
    global model
    #get a pretrained model which can already recognise things like shapes, edges, lines, and objects etc...
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    #replace the final layer so that the model so that the output can be a number corresponding to a folder
    #look at an output from the previous layer containing many different patterns, and squash it into 'number of options' possiple outputs
    model.fc = torch.nn.Linear(model.fc.in_features, len(os.listdir("images")))
    
    model.load_state_dict(torch.load("model.pth", map_location=device))
    #move the model to the gpu if possible
    model.to(device)
    #turns off features such as 'dropout' which are used during training, to make predictions more consistent
    model.eval()

#returns a number corresponding to the folder of the image
def run(path):
    image = PIL.Image.open(path)
    image = transformation(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return round(predicted.item()) 

#get an image from and run the model on it
def get_image():
    path = filedialog.askopenfile(filetypes=[("Image Files", "*.PNG *.JPEG *.JPG *.PNG *.WEBP *.BMP *.TIFF")]).name
    image = PIL.Image.open(path).resize((224, 224))
    #convert the image into a png so that it is compatible with tkinter regardless of the format
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = tkinter.PhotoImage(data=buffer.read())
    image_label.configure(image=image)
    image_label.image = image
    image_label.grid(row=0, column=0)
    result = run(path)
    print(result)
    folders = [name for name in os.listdir("images")]
    result_label.configure(text=folders[result])

#draw a basic GUI onto the screen
def draw():
    load()
    global frame, result_label, image_label
    window = tkinter.Tk()
    window.resizable(0, 0)
    title = ttk.Label(text="Image Detector", padding=20)
    title.grid(row=0, column=0)
    frame = ttk.Frame(width=224, height=224)
    frame.grid_propagate(False)
    frame.grid(row=1, column=0)
    image_label = ttk.Label(frame)
    image_label.grid(row=0, column=0)
    result_label = ttk.Label(text="")
    result_label.grid(row=2, column=0, pady=20)
    select = ttk.Button(text="Select an image", command=get_image)
    select.grid(row=3, column=0, pady=20)
    window.mainloop()

def main():
    if len(sys.argv) > 1:
        match sys.argv[1]:
            case "train":
                if len(sys.argv) > 2:
                    train(int(sys.argv[2]))
                else:
                    train(100)
            case "run":
                draw()
    else:
        draw()
main()