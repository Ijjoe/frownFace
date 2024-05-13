"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import numpy as np
from PIL import Image

img_file_buffer = st.file_uploader('Upload a PNG image', type='png')
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    plt.imshow(img_array)