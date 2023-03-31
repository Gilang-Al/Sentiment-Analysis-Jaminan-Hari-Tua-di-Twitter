import tkinter as tk
import tkinter.font as tkFont
import tweepy
import csv
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import time
import os
from tkinter import ttk
from tkinter import filedialog as fd
from model import all_in_modeling
from test_model import predict
from screping import screping


LARGEFONT =("Verdana", 35)

class tkinterApp(tk.Tk):
    
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)
    
        # creating a container
        container = tk.Frame(self)
        container.pack(side = "top", fill = "both", expand = True)
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
        

        # initializing frames to an empty array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, Page1, Page2):

            frame = F(container, self)

            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame

            frame.grid(row = 0, column = 0, sticky ="nsew")

        self.show_frame(StartPage)

    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
  
  
# first window frame startpage
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        # label of frame Layout 2
        label = ttk.Label(self, text ="Buat Dataset", font = LARGEFONT)
        
        labelmenu = ttk.Label(self, text ="Menu", font = ("Courier", 15))
        # putting the grid in its place by using
        # grid
        label.grid(row = 0, column = 4, padx = 470, pady = 10)
        labelmenu.grid(row = 0, column = 1, padx = 10, pady = 10)

        button1 = ttk.Button(self, text ="Dataset Pribadi", width=25,
        command = lambda : controller.show_frame(Page1))
    
        # putting the button in its place by
        # using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 10, ipady=8)

        ## button to show frame 2 with text layout2
        button2 = ttk.Button(self, text ="Test Model", width=25, 
        command = lambda : controller.show_frame(Page2))
    
        # putting the button in its place by
        # using grid
        button2.grid(row = 2, column = 1, padx = 10, pady = 10, ipady=8)
        
        self.rowconfigure(4, weight=3)
        label3 = ttk.Label(self, text ="")
        label3.grid(row = 3, column = 1, padx = 10, pady = 10, ipady=8, sticky='w')
        label4 = ttk.Label(self, text ="")
        label4.grid(row = 4, column = 1, padx = 10, pady = 10, ipady=8, sticky='w')
        button4 = ttk.Button(self, text ="help ?", width=25, command=self.help)
        button4.grid(row = 5, column = 1, padx = 10, pady = 10, ipady=8, sticky='w')
        
        self.GLineEdit_499 = ttk.Entry(self, text='entry', width=50)
        self.GLineEdit_499.grid(row= 1, column = 4, padx = 10, pady = 10, ipady=6)
        
        bsp = ttk.Button(self, text='buat sentimen', command=self.buat_sentimen, width=50)
        bsp.grid(row= 2, column = 4, padx = 10, pady = 10, ipady=8)
        ttk.Separator(self, orient=tk.VERTICAL).grid(column=2, row=0, rowspan=9, sticky='ns')
        
        self.prog = ttk.Progressbar(self, length=300)
        self.prog.grid(row= 3, column = 4, padx = 10, pady = 10, ipady=8)
            
    def help(self):
        tk.messagebox.showinfo("Help",'Menu "Buat Dataset" digunakan untuk scrapping, preprocessing, labelling data, lalu mengolah data menjadi dataset agar bisa dilakukan modeling. \n1. Masukan kata kunci atau kalimat ke dalam kolom input, lalu tekan buttton "buat sentiment". \n2. Sistem menampilkan grafik hasil labelling, Confusion matrix dan Piechart hasil sentimen \n3. Silahkan simpan grafik tersebut.')

    def buat_sentimen(self):
        search_key = self.GLineEdit_499.get()
        screping(search_key, self.prog, app)  


# second window frame page1
class Page1(tk.Frame):
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="Dataset Pribadi", font = LARGEFONT)
        label.grid(row = 0, column = 4, padx = 445, pady = 10)

        labelmenu = ttk.Label(self, text ="Menu", font = ("Courier", 15))
        labelmenu.grid(row = 0, column = 1, padx = 10, pady = 10)
        # button to show frame 2 with text
        # layout2
        button1 = ttk.Button(self, text ="Buat Dataset", width=25,
                            command = lambda : controller.show_frame(StartPage))
    
        # putting the button in its place
        # by using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 10, ipady=8)

        # button to show frame 2 with text
        # layout2
        button2 = ttk.Button(self, text ="Test Model", width=25,
                            command = lambda : controller.show_frame(Page2))
    
        # putting the button in its place by
        # using grid
        button2.grid(row = 2, column = 1, padx = 10, pady = 10, ipady=8)

        button3 = ttk.Button(self, text ="help ?", width=25, command=self.help)
        button3.grid(row = 5, column = 1, padx = 10, pady = 10, ipady=8, sticky='w')

        model_sendiri = ttk.Button(self, text='Open a File', command=self.select_file, width=50)
        model_sendiri.grid(row= 2, column = 4, padx = 10, pady = 10, ipady=8)
        
        self.rowconfigure(4, weight=3)
        label3 = ttk.Label(self, text ="")
        label3.grid(row = 3, column = 1, padx = 10, pady = 10, ipady=8, sticky='w')
        label4 = ttk.Label(self, text ="")
        label4.grid(row = 4, column = 1, padx = 10, pady = 10, ipady=8, sticky='w')
        
        ttk.Separator(self, orient=tk.VERTICAL).grid(column=2, row=0, rowspan=9, sticky='ns')
    
    def help(self):
        tk.messagebox.showinfo("Help",'Menu "Dataset Pribadi" digunakan untuk input file yang user miliki tanpa melalu menu "Buat Dataset". \n1. Button Open a File untuk mengarahkan user pada file dataset yang disimpan di dictonary user. \n2. Syarat filenya yaitu berbentuk csv.')
        
    def select_file(self):
        filetypes = (
        ('csv files', '*.csv'),
        ('All files', '*.*')
        )

        filename = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)

        print(filename)
        try:
            all_in_modeling(str(filename))
        except:
            tk.messagebox.showinfo("error","data/file tidak ditemukan")


# third window frame page2
class Page2(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="Test model", font = LARGEFONT)
        label.grid(row = 0, column = 4, padx = 493, pady = 10)

        labelmenu = ttk.Label(self, text ="Menu", font = ("Courier", 15))
        labelmenu.grid(row = 0, column = 1, padx = 10, pady = 10)
        # button to show frame 2 with text
        # layout2
        button1 = ttk.Button(self, text ="Dataset Pribadi", width=25,
                            command = lambda : controller.show_frame(Page1))
    
        # putting the button in its place by
        # using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 10, ipady=8)

        # button to show frame 3 with text
        # layout3
        button2 = ttk.Button(self, text ="Buat Dataset", width=25,
                            command = lambda : controller.show_frame(StartPage))
    
        # putting the button in its place by
        # using grid
        button2.grid(row = 2, column = 1, padx = 10, pady = 10, ipady=8)

        button3 = ttk.Button(self, text ="help ?", width=25, command=self.help)
        button3.grid(row = 8, column = 1, padx = 10, pady = 10, ipady=8, sticky='w')
        
        modeling = ttk.Button(self, text='Klasifikasi', command=self.tes_model, width=50)
        modeling.grid(row= 2, column = 4, padx = 10, pady = 10, ipady=8)
        
        self.rowconfigure(7, weight=6)
        label3 = ttk.Label(self, text ="")
        label3.grid(row = 3, column = 1, padx = 10, pady = 10, ipady=8, sticky='w')
        label4 = ttk.Label(self, text ="")
        label4.grid(row = 4, column = 1, padx = 10, pady = 10, ipady=8, sticky='w')
        label5 = ttk.Label(self, text ="")
        label5.grid(row = 5, column = 1, padx = 10, pady = 10, ipady=8, sticky='w')
        label6 = ttk.Label(self, text ="")
        label6.grid(row = 6, column = 1, padx = 10, pady = 10, ipady=8, sticky='w')
        label7 = ttk.Label(self, text ="")
        label7.grid(row = 7, column = 1, padx = 10, pady = 10, ipady=8, sticky='w')
        
        self.tes_model = ttk.Entry(self, text='entry', width=50)
        self.tes_model.grid(row= 1, column = 4, padx = 10, pady = 10, ipady=6)
        
        self.label1 = ttk.Label(self, text ="", font = ("Courier", 20))
        self.label1.grid(row = 4, column = 5, padx = 10, pady = 10)
        self.label2 = ttk.Label(self, text ="", font = ("Courier", 20))
        self.label2.grid(row = 5, column = 5, padx = 10, pady = 10)
        self.label3 = ttk.Label(self, text ="", font = ("Courier", 20))
        self.label3.grid(row = 6, column = 5, padx = 10, pady = 10)
        
        self.label11 = ttk.Label(self, text ="Sentimen:", font = ("Courier", 20))
        self.label11.grid(row = 4, column = 4, padx = 10, pady = 10)
        self.label22 = ttk.Label(self, text ="Positif :", font = ("Courier", 20))
        self.label22.grid(row = 5, column = 4, padx = 10, pady = 10)
        self.label33 = ttk.Label(self, text ="Negatif :", font = ("Courier", 20))
        self.label33.grid(row = 6, column = 4, padx = 10, pady = 10)
        
        ttk.Separator(self, orient=tk.VERTICAL).grid(column=2, row=0, rowspan=9, sticky='ns')
        
    def help(self):
        tk.messagebox.showinfo("Help",'Menu "Test Model" berguna untuk mencoba model yang sudah dibuat pada halaman sebelumnya. \n1. Masukan kata atau kalimat yang ingin dicoba untuk melihat sentimen dari kata atau kalimat tersebut. \n2. Sistem akan menampilakan sentimen dari kata atau kalimat tersebut.')
        
    def tes_model(self):
        # try:
        texts = self.tes_model.get()
        # print(predict(texts))
        pred, neg, pos = predict(texts)
        self.label11.config(text="Sentimen : "+pred)
        self.label22.config(text="Positif : "+pos)
        self.label33.config(text="Negatif : "+neg)
        # except:
        #     tk.messagebox.showinfo("error","data/file tidak ditemukan")
        

# Driver Code
app = tkinterApp()
# app.attributes('-fullscreen',True)
app.mainloop()

