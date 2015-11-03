from Tkinter import *

from xor_model import *

import numpy as np
from PIL import Image
from PIL import ImageTk
from threading import Thread
import time

class GUI(Tk):

    def __init__(self):

        Tk.__init__(self, None)

        self.title('Xor Network')
        self.net = Network(batch_size=1)
        self.inference_step = self.net.build_inference_step()

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
        self.latency.set(0.1)
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

        Label(self, text="eps_y").pack(side=LEFT)
        self.eps_y = DoubleVar()
        self.eps_y.set(0.)
        Entry(self, textvariable=self.eps_y, width=5).pack(side=LEFT)


        # CLAMP BUTTON
        def clamp():
            index = self.index.get()
            self.net.clamp(index=index,clear=True)
            self.update_canvas()
        Button(self, text="Clear", command=clamp).pack(side=LEFT)

        # INFERENCE BUTTON
        def set_inference():
            self.lambda_x.set(1.)
            self.lambda_y.set(0.)
            self.eps_s.set(0.1)
            self.eps_w.set(0.)
            self.eps_y.set(0.)
        Button(self, text="Inference", command=set_inference).pack(side=LEFT)

        # LEARNING BUTTON
        def set_learning():
            self.lambda_x.set(1.)
            self.lambda_y.set(1.)
            self.eps_s.set(0.1)
            self.eps_w.set(0.1)
            self.eps_y.set(0.1)
        Button(self, text="Learning", command=set_learning).pack(side=LEFT)


        Thread(target = self.run).start()

    def update_canvas(self, first_time = False):

        x_data_mat = 256*self.net.x_data.get_value().reshape((1,2))
        x_data_img=Image.fromarray(x_data_mat).resize((50,25))
        self.x_data_imgTk=ImageTk.PhotoImage(x_data_img)

        x_mat = 256*self.net.x.get_value().reshape((1,2))
        x_img=Image.fromarray(x_mat).resize((50,25))
        self.x_imgTk=ImageTk.PhotoImage(x_img)

        h_mat = 256*self.net.h.get_value().reshape((1,3))
        h_img=Image.fromarray(h_mat).resize((75,25))
        self.h_imgTk=ImageTk.PhotoImage(h_img)

        y_mat = 256*self.net.y.get_value().reshape((1,1))
        y_img=Image.fromarray(y_mat).resize((25,25))
        self.y_imgTk=ImageTk.PhotoImage(y_img)

        y_data_mat = 256*self.net.y_data.get_value().reshape((1,1))
        y_data_img=Image.fromarray(y_data_mat).resize((25,25))
        self.y_data_imgTk=ImageTk.PhotoImage(y_data_img)

        bx_mat = self.net.bx.get_value().reshape((1,2))
        W1_mat = self.net.W1.get_value().reshape((2,3))
        bh_mat = self.net.bh.get_value().reshape((1,3))
        W2_mat = self.net.W2.get_value().reshape((1,3))
        by_mat = self.net.by.get_value().reshape((1,1))

        def aux(bx_mat, W1_mat, bh_mat, W2_mat, by_mat):
            maxi = max( abs(np.max(bx_mat)), abs(np.max(W1_mat)), abs(np.max(bh_mat)), abs(np.max(W2_mat)), abs(np.max(by_mat)) )
            bx = 128 + bx_mat / maxi * 128
            W1 = 128 + W1_mat / maxi * 128
            bh = 128 + bh_mat / maxi * 128
            W2 = 128 + W2_mat / maxi * 128
            by = 128 + by_mat / maxi * 128
            return (bx, W1, bh, W2, by)

        (bx_mat, W1_mat, bh_mat, W2_mat, by_mat) = aux(bx_mat, W1_mat, bh_mat, W2_mat, by_mat)
        bx_img=Image.fromarray(bx_mat).resize((50,25))
        self.bx_imgTk=ImageTk.PhotoImage(bx_img)
        W1_img=Image.fromarray(W1_mat).resize((75,50))
        self.W1_imgTk=ImageTk.PhotoImage(W1_img)
        bh_img=Image.fromarray(bh_mat).resize((75,25))
        self.bh_imgTk=ImageTk.PhotoImage(bh_img)
        W2_img=Image.fromarray(W2_mat).resize((75,25))
        self.W2_imgTk=ImageTk.PhotoImage(W2_img)
        by_img=Image.fromarray(by_mat).resize((25,25))
        self.by_imgTk=ImageTk.PhotoImage(by_img)

        if first_time:
            self.y_data_img_canvas         = self.canvas.create_image(300, 50,  image = self.y_data_imgTk)
            self.y_img_canvas              = self.canvas.create_image(300, 150, image = self.y_imgTk)
            self.h_img_canvas              = self.canvas.create_image(300, 250, image = self.h_imgTk)
            self.x_img_canvas              = self.canvas.create_image(300, 350, image = self.x_imgTk)
            self.x_data_img_canvas         = self.canvas.create_image(300, 450, image = self.x_data_imgTk)
            self.W1_img_canvas             = self.canvas.create_image(300, 300, image = self.W1_imgTk)
            self.W2_img_canvas             = self.canvas.create_image(300, 200, image = self.W2_imgTk)
            self.bx_img_canvas             = self.canvas.create_image(450, 350, image = self.bx_imgTk)
            self.bh_img_canvas             = self.canvas.create_image(450, 250, image = self.bh_imgTk)
            self.by_img_canvas             = self.canvas.create_image(450, 150, image = self.by_imgTk)
        else:
            self.canvas.itemconfig(self.y_data_img_canvas,         image = self.y_data_imgTk)
            self.canvas.itemconfig(self.y_img_canvas,              image = self.y_imgTk)
            self.canvas.itemconfig(self.h_img_canvas,              image = self.h_imgTk)
            self.canvas.itemconfig(self.x_img_canvas,              image = self.x_imgTk)
            self.canvas.itemconfig(self.x_data_img_canvas,         image = self.x_data_imgTk)
            self.canvas.itemconfig(self.W1_img_canvas,             image = self.W1_imgTk)
            self.canvas.itemconfig(self.W2_img_canvas,             image = self.W2_imgTk)
            self.canvas.itemconfig(self.bx_img_canvas,             image = self.bx_imgTk)
            self.canvas.itemconfig(self.bh_img_canvas,             image = self.bh_imgTk)
            self.canvas.itemconfig(self.by_img_canvas,             image = self.by_imgTk)

    def run(self):

        while True:

            while self.running:

                index = self.index.get() # index of the test example in the test set
                self.net.clamp(index=index, clear=False)

                lambda_x = self.lambda_x.get()
                lambda_y = self.lambda_y.get()
                eps_s = self.eps_s.get()
                eps_w = self.eps_w.get()
                eps_y = self.eps_y.get()

                [energy, prediction, error_rate, square_loss] = self.inference_step(lambda_x, lambda_y, eps_s, eps_w, eps_y)

                print("energy = %f, pred = %f, error = %f, loss = %f" % (energy, prediction, error_rate, square_loss))
                
                self.update_canvas()
                time.sleep(self.latency.get())

            time.sleep(0.2)

if __name__ == "__main__":

    GUI().mainloop()