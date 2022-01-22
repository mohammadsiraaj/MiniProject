
import streamlit as st
import numpy as np
from PIL import Image 
from tensorflow.keras.models import load_model
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

#Loading the model
def loading_model():
  fp = "mymodel150.h5"
  model_loader = load_model(fp)
  return model_loader

cnn = loading_model()
st.write("""# X-Ray Classification [Pneumonia/Normal]""")
temp = st.file_uploader("Upload X-Ray Image to find the respective class")
buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))
if buffer is None:
  st.text(" ")
else:
    myimg = image.load_img(temp_file.name, target_size=(150, 150),color_mode='grayscale')
    
    # Preprocessing the image
    convertedimg = image.img_to_array(myimg)
    convertedimg = convertedimg/255
    convertedimg = np.expand_dims(convertedimg, axis=0)
    
    #predict
    preds= cnn.predict(convertedimg)
    if preds>= 0.5:
      out = ('{:.1%} percent confirmed that this is a Pneumonia case'.format(preds[0][0]))
      
    else: 
      out = ('{:.1%} percent confirmed that this is a Normal case'.format(1-preds[0][0]))
    st.success(out)
      
    image = Image.open(temp)
    st.image(image,use_column_width=True)
          
            

  

  