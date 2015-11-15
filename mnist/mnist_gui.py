from Tkinter import *

from mnist_model import *

import numpy as np
from PIL import Image
from PIL import ImageTk
from threading import Thread
import time

class GUI(Tk):

    def __init__(self):

        Tk.__init__(self, None)

        self.title('Biological Network for MNIST')
        self.net = Network(batch_size=1)

        self.canvas = Canvas(self, width=600, height=500)
        self.canvas.pack(side=BOTTOM)

        # FREQUENCY OF UPDATES
        self.latency = .1

        # INDEX OF TEST EXAMPLE (IN THE TRAINING SET)
        Label(self, text="image").pack(side=LEFT)
        self.index = StringVar()
        self.index.set("0")
        Entry(self, textvariable=self.index, width=5).pack(side=LEFT)


        # PARAMETERS OF THE ITERATIVE PROCEDURE

        Label(self, text="lambda_x").pack(side=LEFT)
        self.lambda_x = DoubleVar()
        self.lambda_x.set(1.)
        Entry(self, textvariable=self.lambda_x, width=5).pack(side=LEFT)

        Label(self, text="lambda_y").pack(side=LEFT)
        self.lambda_y = DoubleVar()
        self.lambda_y.set(0.)
        Entry(self, textvariable=self.lambda_y, width=5).pack(side=LEFT)

        Label(self, text="eps_x").pack(side=LEFT)
        self.eps_x = DoubleVar()
        self.eps_x.set(.1)
        Entry(self, textvariable=self.eps_x, width=5).pack(side=LEFT)

        Label(self, text="eps_h").pack(side=LEFT)
        self.eps_h = DoubleVar()
        self.eps_h.set(.1)
        Entry(self, textvariable=self.eps_h, width=5).pack(side=LEFT)

        Label(self, text="eps_y").pack(side=LEFT)
        self.eps_y = DoubleVar()
        self.eps_y.set(.1)
        Entry(self, textvariable=self.eps_y, width=5).pack(side=LEFT)



        [self.energy, self.norm_grad, self.prediction, _, self.mse, _, _] = self.net.iterate(lambda_x=0., lambda_y=0., epsilon_x=0., epsilon_h=0., epsilon_y=0., alpha_W1=0., alpha_W2=0.)

        self.update_canvas(first_time=True)

        Thread(target = self.run).start()

    def update_canvas(self, first_time = False):

        x_data_mat = 256*self.net.outside_world.x_data.get_value().reshape((28,28))
        x_data_img=Image.fromarray(x_data_mat).resize((140,140))
        self.x_data_imgTk=ImageTk.PhotoImage(x_data_img)

        x_mat = 256*self.net.x.get_value().reshape((28,28))
        x_img=Image.fromarray(x_mat).resize((140,140))
        self.x_imgTk=ImageTk.PhotoImage(x_img)

        h_mat = 256*self.net.h.get_value().reshape((10,self.net.n_hidden/10))
        h_img=Image.fromarray(h_mat).resize((self.net.n_hidden/2,50))
        self.h_imgTk=ImageTk.PhotoImage(h_img)

        y_mat = 256*self.net.y.get_value().reshape((1,10))
        y_img=Image.fromarray(y_mat).resize((250,25))
        self.y_imgTk=ImageTk.PhotoImage(y_img)

        y_data_one_hot_mat = np.zeros( shape=(1, 10) )
        index = self.net.outside_world.y_data.get_value()[0]
        y_data_one_hot_mat[0,index] = 256.
        y_data_one_hot_img=Image.fromarray(y_data_one_hot_mat).resize((250,25))
        self.y_data_one_hot_imgTk=ImageTk.PhotoImage(y_data_one_hot_img)

        if first_time:
            self.y_data_one_hot_img_canvas = self.canvas.create_image(400, 50,  image = self.y_data_one_hot_imgTk)
            self.y_img_canvas              = self.canvas.create_image(400, 100, image = self.y_imgTk)
            self.h_img_canvas              = self.canvas.create_image(400, 150, image = self.h_imgTk)
            self.x_img_canvas              = self.canvas.create_image(400, 250, image = self.x_imgTk)
            self.x_data_img_canvas         = self.canvas.create_image(400, 400, image = self.x_data_imgTk)
            self.energy_canvas             = self.canvas.create_text(  20, 100, anchor=W, font="Purisa", text  = "Energy = %.1f"        % (self.energy))
            self.norm_grad_canvas          = self.canvas.create_text(  20, 200, anchor=W, font="Purisa", text  = "Norm Gradient = %.1f" % (self.norm_grad))
            self.prediction_canvas         = self.canvas.create_text(  20, 300, anchor=W, font="Purisa", text  = "Prediction = %i"      % (self.prediction[0]))
            self.mse_canvas                = self.canvas.create_text(  20, 400, anchor=W, font="Purisa", text  = "Squared Error = %.4f" % (self.mse))
        else:
            self.canvas.itemconfig(self.y_data_one_hot_img_canvas, image = self.y_data_one_hot_imgTk)
            self.canvas.itemconfig(self.y_img_canvas,              image = self.y_imgTk)
            self.canvas.itemconfig(self.h_img_canvas,              image = self.h_imgTk)
            self.canvas.itemconfig(self.x_img_canvas,              image = self.x_imgTk)
            self.canvas.itemconfig(self.x_data_img_canvas,         image = self.x_data_imgTk)
            self.canvas.itemconfig(self.energy_canvas,             text  = "Energy = %.1f"        % (self.energy))
            self.canvas.itemconfig(self.norm_grad_canvas,          text  = "Norm Gradient = %.1f" % (self.norm_grad))
            self.canvas.itemconfig(self.prediction_canvas,         text  = "Prediction = %i"      % (self.prediction[0]))
            self.canvas.itemconfig(self.mse_canvas,                text  = "Squared Error = %.4f" % (self.mse))

    def run(self):

        while True:

            index = self.index.get() # index of the test example in the test set
            if index.isdigit():
                index = int(index)
            index = hash(index)
            index = index % 10000
            self.net.outside_world.set_test(index=index)

            lambda_x = np.float32(self.lambda_x.get())
            lambda_y = np.float32(self.lambda_y.get())
            eps_x = np.float32(self.eps_x.get())
            eps_h = np.float32(self.eps_h.get())
            eps_y = np.float32(self.eps_y.get())

            [self.energy, self.norm_grad, self.prediction, _, self.mse, _, _] = self.net.iterate(lambda_x=lambda_x, lambda_y=lambda_y, epsilon_x=eps_x, epsilon_h=eps_h, epsilon_y=eps_y, alpha_W1=0., alpha_W2=0.)
            
            self.update_canvas()
            time.sleep(self.latency)

if __name__ == "__main__":

    GUI().mainloop()