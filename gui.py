from Tkinter import *

from biologicalnetwork import *

import numpy as np
from PIL import Image
from PIL import ImageTk
from threading import Thread
import time

class GUI(Tk):

    def __init__(self):

        Tk.__init__(self, None)

        self.title('Biological Network')
        self.net = Network(batch_size=1)
        self.inference_step = self.net.build_inference_step()

        train_set, valid_set, test_set = mnist()
        (self.test_set_x, self.test_set_y) = test_set

        self.canvas = Canvas(self, width=600, height=500)
        self.canvas.pack(side=BOTTOM)

        self.update_canvas(first_time=True)

        # START BUTTON
        self.running = False
        def onClickStartButton():
            self.running = not self.running
            if self.running:
                startButton.configure(text="Stop")
            else:
                startButton.configure(text="Start")
        startButton = Button(self, text="Start", command=onClickStartButton)
        startButton.pack(side=LEFT)

        # FREQUENCY OF UPDATES
        Label(self, text="latency").pack(side=LEFT)
        self.latency = DoubleVar()
        self.latency.set(1)
        Entry(self, textvariable=self.latency, width=5).pack(side=LEFT)

        # INDEX OF TEST EXAMPLE IN THE TEST SET
        Label(self, text="image").pack(side=LEFT)
        self.index = IntVar()
        self.index.set(0)
        Entry(self, textvariable=self.index, width=5).pack(side=LEFT)


        # INFERENCE PARAMETERS

        Label(self, text="lambda_x").pack(side=LEFT)
        self.lambda_x = DoubleVar()
        self.lambda_x.set(1.)
        Entry(self, textvariable=self.lambda_x, width=5).pack(side=LEFT)

        Label(self, text="lambda_y").pack(side=LEFT)
        self.lambda_y = DoubleVar()
        self.lambda_y.set(0.)
        Entry(self, textvariable=self.lambda_y, width=5).pack(side=LEFT)

        Label(self, text="eps_s").pack(side=LEFT)
        self.eps_s = DoubleVar()
        self.eps_s.set(0.1)
        Entry(self, textvariable=self.eps_s, width=5).pack(side=LEFT)

        Label(self, text="eps_w").pack(side=LEFT)
        self.eps_w = DoubleVar()
        self.eps_w.set(0.)
        Entry(self, textvariable=self.eps_w, width=5).pack(side=LEFT)


        # CLEAR BUTTON
        def clear():
            self.net.clear()
            self.update_canvas()
        Button(self, text="Clear", command=clear).pack(side=LEFT)

        # INFERENCE BUTTON
        def set_inference():
            self.lambda_x.set(1.)
            self.lambda_y.set(0.)
            self.eps_s.set(0.1)
            self.eps_w.set(0.)
        Button(self, text="Inference", command=set_inference).pack(side=LEFT)

        # LEARNING BUTTON
        def set_learning():
            self.lambda_x.set(1.)
            self.lambda_y.set(0.5)
            self.eps_s.set(0.1)
            self.eps_w.set(0.1)
        Button(self, text="Learning", command=set_learning).pack(side=LEFT)


        Thread(target = self.run).start()

    def update_canvas(self, first_time = False):

        x_mat = 256*self.net.x.get_value().reshape((28,28))
        x_img=Image.fromarray(x_mat).resize((140,140))
        self.x_imgTk=ImageTk.PhotoImage(x_img)

        h_mat = 256*self.net.h.get_value().reshape((10,50))
        h_img=Image.fromarray(h_mat).resize((250,50))
        self.h_imgTk=ImageTk.PhotoImage(h_img)

        y_mat = 256*self.net.y.get_value().reshape((1,10))
        y_img=Image.fromarray(y_mat).resize((250,25))
        self.y_imgTk=ImageTk.PhotoImage(y_img)

        if first_time:
            self.y_img_canvas = self.canvas.create_image(300, 100, image = self.y_imgTk)
            self.h_img_canvas = self.canvas.create_image(300, 200, image = self.h_imgTk)
            self.x_img_canvas = self.canvas.create_image(300, 300, image = self.x_imgTk)
        else:
            self.canvas.itemconfig(self.y_img_canvas, image = self.y_imgTk)
            self.canvas.itemconfig(self.h_img_canvas, image = self.h_imgTk)
            self.canvas.itemconfig(self.x_img_canvas, image = self.x_imgTk)

    def run(self):

        while True:

            while self.running:

                index = self.index.get() # index of the test example in the test set
                x_data = self.test_set_x[index:index+1,]
                y_data = self.test_set_y[index:index+1,]

                lambda_x = self.lambda_x.get()
                lambda_y = self.lambda_y.get()
                eps_s = self.eps_s.get()
                eps_w = self.eps_w.get()

                [energy, prediction, error] = self.inference_step(x_data, y_data, lambda_x, lambda_y, eps_s, eps_w)

                print("energy = %f" % (energy))
                
                self.update_canvas()
                time.sleep(self.latency.get())

            time.sleep(0.2)

if __name__ == "__main__":

    GUI().mainloop()