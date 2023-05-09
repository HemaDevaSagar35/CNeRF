import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import *
from PIL import Image, ImageTk
import sys
sys.path.insert(0, 'C:\\Users\\phdsa\\Documents\\Computer_Graphics\\project_dev\\CNeRF')
import inference_app

class VanillaApp(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()
        self.photo = None
        self.mask = None

        self.components = {}

        self.photo_3d = None
        self.mask_3d = None 
        # frame1 = LabelFrame(master, text="Original", padx=15, pady=15)
        # frame1.grid(row=0, column=0)
        panedwindow=tk.PanedWindow(master, bd=4, relief = 'raised', bg = 'red', orient=HORIZONTAL)
        panedwindow.pack(fill=BOTH, expand=True)
        frame1=tk.Frame(panedwindow,width=300,height=300, relief=SUNKEN)
        # frame2=tk.Frame(panedwindow,width=300,height=300, relief=SUNKEN)
        frame3=tk.Frame(panedwindow,width=500,height=500, relief=SUNKEN)
        # no_vanilla_image = tk.Button(frame1, text = "Generate Image", command = lambda: self.call_back("Here is Vanilla!"))

        no_vanilla_image = tk.Button(frame1, text = "Generate Image", command = lambda: self.call_back_image("sample.png"))
        no_vanilla_image.place(x=130, y= 10)


        # vanilla_button_two = tk.Button(frame2, text = "Modified Image", command=lambda: self.call_back_modified())
        # vanilla_button_two.pack()

        vanilla_button = tk.Button(frame3, text = "3D View", command=lambda: self.call_back_modified())
        vanilla_button.place(x=400, y= 70)

        self.slider = tk.Scale(frame3, from_=1, to = 10, orient='horizontal')
        self.slider.place(x=250, y= 60)

        nose_button = tk.Button(frame3, text = "Nose", command=lambda: self.call_back_sem_nose())
        nose_button.place(x = 400, y = 160)

        self.slider_nose = tk.Scale(frame3, from_=-50, to = 50, orient='horizontal')
        self.slider_nose.place(x=250, y= 150)

        mouth_button = tk.Button(frame3, text = "Mouth", command=lambda: self.call_back_sem_mouth())
        mouth_button.place(x = 400, y = 250)

        self.slider_mouth = tk.Scale(frame3, from_=-50, to = 50, orient='horizontal')
        self.slider_mouth.place(x=250, y= 240)

        hair_button = tk.Button(frame3, text = "Hair", command=lambda: self.call_back_sem_hair())
        hair_button.place(x = 400, y = 340)

        self.slider_hair = tk.Scale(frame3, from_=-50, to = 50, orient='horizontal')
        self.slider_hair.place(x=250, y= 330)

        eyes_button = tk.Button(frame3, text = "Eyes", command=lambda: self.call_back_sem_eyes())
        eyes_button.place(x = 400, y = 430)

        self.slider_eyes = tk.Scale(frame3, from_=-50, to = 50, orient='horizontal')
        self.slider_eyes.place(x=250, y= 420)


        part_map_button = tk.Button(frame1, text="Semantic Maps", command=lambda: self.call_back_sem_map())
        part_map_button.place(x=130, y= 210)

        self.canvas = Canvas(frame1, width = 400, height=150)
        self.canvas.place(x = 0, y = 50)

        self.canvas_3 = Canvas(frame1, width = 400, height=300)
        self.canvas_3.place(x = 0, y = 240)

        self.canvas_2 = Canvas(frame3, width=240, height=500)
        self.canvas_2.place(x = 0, y = 30)

        # slider_curr_value = tk.DoubleVar()
        

        panedwindow.add(frame1)
        # panedwindow.add(frame2)
        panedwindow.add(frame3)

        #load model
        print("loading the model")
        self.model = inference_app.InferenceObj()
        self.model.load_model()
        print("loading the model - DONE")
       
        # # table_label = tk.Label(master, text="Do you want vanilla ice?")
        # # table_label.pack()
        # # vanilla_button = tk.Button(master, text = "I want Vanilla", command=lambda: self.call_back("Here is Vanilla!"))
        # # vanilla_button.pack()
        # # no_vanilla_button = tk.Button(master, text = "I want something else", command=lambda: self.call_back("Here is bread!"))
        # # no_vanilla_button.pack()
        # no_vanilla_image = tk.Button(master, text = "Generate Image", command = lambda: self.call_back_image("sample.png"))
        # no_vanilla_image.pack()
        
    def call_back(self, message):
        showinfo("This is an Infobox", message)\
    
    def call_back_image(self, path):
        # self.photo = ImageTk.PhotoImage(Image.open(path).resize((64, 64)))
        # self.canvas.create_image(20,20, anchor=NW, image = self.photo)

        self.model.random_input()
        self.model.gen_styles()
        self.model.gen_style_global(self.model.styles)
        self.model.cal_direction()
        img, seg, _ = self.model.output(5)

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(100,20, anchor=NW, image = self.photo)

        self.mask = ImageTk.PhotoImage(seg)
        self.canvas.create_image(100 + 84, 20, anchor=NW, image = self.mask)
        # return "Done"

    def call_back_modified(self):
        
        value = self.slider.get()
        img, seg, _ = self.model.output(value-1)
        self.photo_3d = ImageTk.PhotoImage(img)
        
        self.canvas_2.create_image(20, 20, anchor=NW, image = self.photo_3d)

        self.mask_3d = ImageTk.PhotoImage(seg)
        self.canvas_2.create_image(104, 20, anchor=NW, image = self.mask_3d)


        # print(value)
        # showinfo("This is an Infobox", value)
    
    def call_back_sem_map(self):
        #1 skin/face
        #2 nose
        #3 glass?
        #4 - eyes
        #6 - eye brow - 5
        #8 - ear - 6
        #10 - mouth - 7
        #13 - hair - 8
        #14 - hat - 9
        sem_list = {'face': 1,'nose': 2, 'glass': 3, 'eyes': 4, 'brow': 5, 'ear' : 6, 
                    'mouth' : 7, 'hair' : 8, 'hat' : 9}
        labels = ['face', 'nose', 'glass', 'eyes', 'brow', 'ear', 'mouth', 'hair', 'hat']
        i = 60
        j = 10
        count = 0
        for lab in labels:
            _, _, img_sem = self.model.output(5, [sem_list[lab]])
            self.components[lab] = ImageTk.PhotoImage(img_sem)
            self.canvas_3.create_text(i+35, j+5, text = lab)
            self.canvas_3.create_image(i, j+15, anchor = NW, image = self.components[lab])
            # self.canvas_3.create_text()
            i = i + 84
            count += 1
            if count == 3:
                j = j + 84
                i = 60
                count = 0


    def call_back_sem_nose(self):
        value = self.slider_nose.get()
        img, seg = self.model.output_manipulate(2, value)
        self.photo_nose = ImageTk.PhotoImage(img)
        self.seg_nose = ImageTk.PhotoImage(seg)

        self.canvas_2.create_image(20, 120, anchor = NW, image = self.photo_nose)
        self.canvas_2.create_image(104, 120, anchor = NW, image = self.seg_nose)
    
    def call_back_sem_mouth(self):
        value = self.slider_mouth.get()
        img, seg = self.model.output_manipulate(7, value)
        self.photo_mouth = ImageTk.PhotoImage(img)
        self.seg_mouth = ImageTk.PhotoImage(seg)

        self.canvas_2.create_image(20, 220, anchor = NW, image = self.photo_mouth)
        self.canvas_2.create_image(104, 220, anchor = NW, image = self.seg_mouth)

    
    def call_back_sem_hair(self):
        value = self.slider_hair.get()
        img, seg = self.model.output_manipulate(8, value)
        self.photo_hair = ImageTk.PhotoImage(img)
        self.seg_hair = ImageTk.PhotoImage(seg)

        self.canvas_2.create_image(20, 320, anchor = NW, image = self.photo_hair)
        self.canvas_2.create_image(104, 320, anchor = NW, image = self.seg_hair)
    
    def call_back_sem_eyes(self):
        value = self.slider_eyes.get()
        img, seg = self.model.output_manipulate(4, value)
        self.photo_eyes = ImageTk.PhotoImage(img)
        self.seg_eyes = ImageTk.PhotoImage(seg)

        self.canvas_2.create_image(20, 420, anchor = NW, image = self.photo_eyes)
        self.canvas_2.create_image(104, 420, anchor = NW, image = self.seg_eyes)




# instantiate a VanillaApp object
if __name__ == "__main__":
    VanillaApp().mainloop()