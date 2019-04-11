
import matplotlib.pyplot as plt

# Imports specifically so we can render outputs in Jupyter.
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

class Notebook(object):    
    
    frames = []        

    @staticmethod
    def setup():
        Notebook.frames = []
 
    @staticmethod
    def record(env):
        Notebook.frames.append(env.render(mode = 'rgb_array'))

    @staticmethod
    def replay():
        """
        Displays a list of frames as a gif, with controls
        """
        plt.figure(figsize=(Notebook.frames[0].shape[1] / 72.0, Notebook.frames[0].shape[0] / 72.0), dpi = 72)
        patch = plt.imshow(Notebook.frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(Notebook.frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(Notebook.frames), interval=50)
        display(display_animation(anim, default_mode='loop'))


    @staticmethod
    def log_progress(sequence, every=None, size=None, name='Episodes'):
        from ipywidgets import IntProgress, HTML, VBox
        from IPython.display import display

        is_iterator = False
        if size is None:
            try:
                size = len(sequence)
            except TypeError:
                is_iterator = True
        if size is not None:
            if every is None:
                if size <= 200:
                    every = 1
                else:
                    every = int(size / 200)     # every 0.5%
        else:
            assert every is not None, 'sequence is iterator, set every'

        if is_iterator:
            progress = IntProgress(min=0, max=1, value=1)
            progress.bar_style = 'info'
        else:
            progress = IntProgress(min=0, max=size, value=0)
        label = HTML()
        box = VBox(children=[label, progress])
        display(box)

        index = 0
        try:
            for index, record in enumerate(sequence, 1):
                if index == 1 or index % every == 0:
                    if is_iterator:
                        label.value = '{name}: {index} / ?'.format(
                            name=name,
                            index=index
                        )
                    else:
                        progress.value = index
                        label.value = u'{name}: {index} / {size}'.format(
                            name=name,
                            index=index,
                            size=size
                        )
                yield record
        except:
            progress.bar_style = 'danger'
            raise
        else:
            progress.bar_style = 'success'
            progress.value = index
            label.value = "{name}: {index}".format(
                name=name,
                index=str(index or '?')
            )

