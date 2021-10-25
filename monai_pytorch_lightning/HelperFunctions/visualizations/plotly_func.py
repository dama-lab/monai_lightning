#!/usr/bin/env python3
#%% to plot in browser using plotly
import numpy as np
import plotly.graph_objects as go

#%% show plotly
def show_plotly(fig, render="html"):
  '''Call plotly figure using: fig.show()
  rendering option: "html" (default),"server"
  - option "html": in a local temp html file 
  - option "server": in broswer as a local server:
  > Ref: https://stackoverflow.com/questions/35315726/plotly-how-to-display-charts-in-spyder
  '''
  if render == "html":
    from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
    init_notebook_mode()
    plot(fig)
  elif render == "server":
    import plotly.io as pio
    pio.renderers.default='browser'
    fig.show()


#%% visualizing volumes
def show_volume_plotly(vol, opacity=0.1, isomin=0.1, isomax=0.8, surface_count=17,
                       show=True, render="html"):
  x,y,z = vol.shape
  X, Y, Z = np.mgrid[range(1,x), range(1,y), range(1,z)]
  fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=vol.flatten(),
    isomin=isomin,
    isomax=isomax,
    opacity=opacity, # needs to be small to see through all surfaces
    surface_count=surface_count, # needs to be a large number for good volume rendering
    ))
  if show is True:
    show_plotly(fig, render=render)
  return fig

#%% slicing through volumes
def frame_args(duration):
  ''' for function: "show_volume_plotly" '''
  return {
          "frame": {"duration": duration},
          "mode": "immediate",
          "fromcurrent": True,
          "transition": {"duration": duration, "easing": "linear"},
          }
#%%
def slice_volume_plotly(vol, nb_frames=16, colorscale='Gray',
                       cmin=None, cmax=None, width=600, height=600,
                       title='Slices in volumetric data',
                       show=True, render="html"):
  '''slicing through last dimension
  Ref 1: https://plotly.com/python/visualizing-mri-volume-slices/'''
  # put last dimention to first
  volume = vol.T.astype(np.float32)
  # row, colume numbers
  r, c = volume[0].shape
  if cmin is None: cmin = volume.min()
  if cmax is None: cmax = volume.max()
  # define frames
  fig = go.Figure(frames=[go.Frame(data=go.Surface(
      z=(6.7 - k * 0.1) * np.ones((r, c)),
      surfacecolor=np.flipud(volume[nb_frames - 1 - k]),
      cmin=cmin, cmax=cmax
      ),
      name=str(k) # you need to name the frame for the animation to behave properly
      )
      for k in range(nb_frames)])
  
  # Add data to be displayed before animation starts
  fig.add_trace(go.Surface(
      z=6.7 * np.ones((r, c)),
      surfacecolor=np.flipud(volume[nb_frames-1]),
      colorscale=colorscale,
      cmin=cmin, cmax=cmax,
      colorbar=dict(thickness=20, ticklen=4)
      ))
  
  sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]
  
  # Layout
  fig.update_layout(
         title=title,
         width=width,
         height=height,
         scene=dict(
                    zaxis=dict(range=[-0.1, 6.8], autorange=False),
                    aspectratio=dict(x=1, y=1, z=1),
                    ),
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
         ],
         sliders=sliders
         )
  
  if show is True:
    show_plotly(fig, render=render)
    
  return fig
# Ref 2: https://plotly.com/python/3d-volume-plots/


