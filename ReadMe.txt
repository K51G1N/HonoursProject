These are all the python libraries that I had installed when running my code

I believe the essentials are
ducc0
numpy
matplot
all scikit libraries
___________________________________________________________________
absl-py==0.11.0
astunparse==1.6.3
async-generator==1.10
attrs==20.3.0
bcrypt==3.2.0
bleach==3.3.0
cachetools==4.2.1
capstone==4.0.2
certifi==2020.12.5
cffi==1.14.5
chardet==4.0.0
chart-studio==1.1.0
click==7.1.2
colored-traceback==0.3.0
crypto==1.4.1
cryptography==3.4.7
cycler==0.10.0
Cython==0.29.22
decorator==4.4.2
defusedxml==0.6.0
dnspython==1.16.0
ducc0==0.9.0
entrypoints==0.3
evdev==1.4.0
eventlet==0.31.0
Flask==1.1.2
flatbuffers==1.12
gast==0.3.3
google-auth==1.27.0
google-auth-oauthlib==0.4.2
google-pasta==0.2.0
greenlet==1.1.0
grpcio==1.32.0
h5py==2.10.0
idna==2.10
imageio==2.9.0
imbalanced-learn==0.8.0
imblearn==0.0
imutils==0.5.4
intervaltree==3.1.0
ipython-genutils==0.2.0
itsdangerous==1.1.0
Jinja2==2.11.3
joblib==1.0.1
jsonschema==3.2.0
jupyter-client==6.1.11
jupyter-core==4.7.1
jupyterlab-pygments==0.1.2
kaggle==1.5.10
Keras==2.4.3
Keras-Preprocessing==1.1.2
kiwisolver==1.3.1
llvmlite==0.35.0
Mako==1.1.4
Markdown==3.3.4
MarkupSafe==1.1.1
matplotlib==3.3.4
mistune==0.8.4
MouseInfo==0.1.3
Naked==0.1.31
nbclient==0.5.3
nbconvert==6.0.7
nbformat==5.1.2
nest-asyncio==1.5.1
networkx==2.5
numpy==1.19.5
oauthlib==3.1.0
opencv-contrib-python==4.5.1.48
opt-einsum==3.3.0
packaging==20.9
pandas==1.2.3
pandocfilters==1.4.3
paramiko==2.7.2
Pillow==8.1.1
plotly==4.14.3
plumbum==1.7.0
progressbar2==3.53.1
protobuf==3.15.3
psutil==5.8.0
pwn==1.0
pwntools==4.5.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
PyAutoGUI==0.9.52
pycparser==2.20
pyelftools==0.27
PyGetWindow==0.0.9
Pygments==2.8.0
PyMsgBox==1.0.9
PyNaCl==1.4.0
pynput==1.7.3
pyparsing==2.4.7
pyperclip==1.8.2
pypng==0.0.20
PyRect==0.1.4
pyrsistent==0.17.3
PyScreeze==0.1.27
pyserial==3.5
PySocks==1.7.1
python-dateutil==2.8.1
python-slugify==4.0.1
python-utils==2.5.6
python-xlib==0.29
python3-xlib==0.15
PyTweening==1.0.3
pytz==2021.1
PyWavelets==1.1.1
PyYAML==5.4.1
pyzmq==22.0.3
QtPy==1.9.0
requests==2.25.1
requests-oauthlib==1.3.0
retrying==1.3.3
ROPGadget==6.5
rpyc==5.0.1
rsa==4.7.2
scikit-image==0.18.1
scikit-learn==0.24.1
scikit-plot==0.3.7
scipy==1.6.1
seaborn==0.11.1
shellescape==3.8.1
six==1.15.0
sortedcontainers==2.3.0
tensorboard==2.4.1
tensorboard-plugin-wit==1.8.0
tensorflow-cpu==2.4.1
tensorflow-estimator==2.4.0
termcolor==1.1.0
testpath==0.4.4
text-unidecode==1.3
threadpoolctl==2.1.0
tifffile==2021.2.26
tornado==6.1
tqdm==4.58.0
traitlets==5.0.5
typing-extensions==3.7.4.3
unicorn==1.0.2rc3
urllib3==1.26.3
webencodings==0.5.1
Werkzeug==1.0.1
Whirlpool==1.0.0
wrapt==1.12.1
xgboost==1.4.1
___________________________________________________________________

Create a new Folder: e.g 17E4476
In this folder place all the python files:
    - SourceSystem.py
    - DataAcquisition.py
    - keagan_first_sim.npy
    - Snippy.py
    - SnipPS.py
    - MultiModel.py
    - MultiModel_Test.py
    -
In this folder create 4 subfolders:
    -Experiment1
    -Experiment2
    -Experiment3
    -data
Inside of data place the uvw.npy file

You may then run the SourceSystem.py file

It will then ask you for the absolute path to Experiment 1, give the absolute path you created for the Experiment1 folder.
Repeat like wise when prompted for the absolute paths of Experiment 2 and 3.

The program will then proceed to Generate all the Seen Train/Test data, the Unseen Test data, 4
Train the model using the Seen data and finally Test the model using the unseen data.
It will do this for all three experiments. 
It will take a while to get through everything.

All the results and diagrams are in their folders.


