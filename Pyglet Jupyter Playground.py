
# coding: utf-8

# # Pyglet + Jupyter Inline?
# It's maybe possible to make Pyglet graphics display inline? I read [this google groups post](https://groups.google.com/forum/#!topic/pyglet-users/Ma04eKqBwlE) talking about adding Pyglet support, and it looks like that [found its way to IPython](https://github.com/ipython/ipython/pull/743)? And there is some Pyglet code [still around](https://github.com/ipython/ipython/search?q=pyglet).
# 
# But, for all of that, I'm not sure how to make it work. It seems like `%gui pyglet` is not supported.

# In[1]:

get_ipython().magic(u'gui pyglet')


# In[2]:

import pyglet


# In[3]:

window = pyglet.window.Window(display=None)
window.on_close = lambda:window.close()
label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')


# In[4]:

def draw_triangle():
    pyglet.gl.glBegin(pyglet.gl.GL_TRIANGLES)
    for p in [(20,30), (200,100), (100,200)]:
        pyglet.gl.glVertex3f(p[0], p[1],0)  # draw each vertex
    pyglet.gl.glEnd()


# In[ ]:




# In[5]:

for _ in range(200):
    window.clear()
    window.switch_to()
    window.dispatch_events()

    label.draw()
    draw_triangle()
    
    window.flip()


# In[ ]:




# On my Mac, this opens a separate Cocoa window through the Python/Idle program, and displays the window:
# <img src="pyglet-screenshot.png">
# 
# But I would like it to display inline, here, so that you can see the window when running Jupyter on a server, such as the MyBinder server.

# In[ ]:



