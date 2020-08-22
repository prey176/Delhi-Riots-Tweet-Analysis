#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
from tkinter import ttk
import import_ipynb
import backend

root=Tk()
mainframe = ttk.Frame(root, padding="20 20 24 24")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)


variable1=StringVar() # Value saved here

def search():
    url = variable1.get()
    result = 'Error'
    try:
        result = backend.solve(url)
    except:
        pass
    print (result)
    labelvar.set(result)
    return ''

ttk.Entry(mainframe, width=7, textvariable=variable1).grid(column=2, row=1)

ttk.Label(mainframe, text="label").grid(column=1, row=1)

ttk.Button(mainframe, text="Search", command=search).grid(column=2, row=13)

ttk.Label(mainframe, text="Output :").grid(column=1, row=25)
labelvar = StringVar()
labelvar.set("None")
ttk.Label(mainframe, textvariable = labelvar).grid(column=3, row=25)



root.mainloop()


# In[ ]:




