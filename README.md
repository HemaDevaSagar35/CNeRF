# CNeRF
model can be found [here](https://drive.google.com/file/d/11yghQMlpJk17RqCq9Q873jY_3PzXcVKA/view?usp=share_link)
create a folder called checkpoints and put the model in there
Below are the instruction to run the application

1. Make sure you have pytorch and pytorch 3D. For Pytorch 3D you have to do source install. You can find relevant detail [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).Note Installing this is a very long process. Like compiling the source code takes close to 15 - 20 minutes. Please be patient.
2. Do pip install -r requirement.txt
3. Go to the app folder and run the application as \
   ```python app.py```

If you want to do training from scratch, you have to run the command \
```python train_CNeRF_modified.py```

But before running this command, you have to adjust the dataset directory parameter dataset_path and checkpoints_dir parameter in options.py

Also you can find the dataset I used [here](https://drive.google.com/drive/folders/1cgYnhTgxmzvmR-cPjT8Hy9RAMA56sW3y?usp=share_link)

[Project link](https://hemadevasagar35.github.io/): For more details check the project details webpage.

References\
[1] https://github.com/MrTornado24/FENeRF \
[2] https://github.com/TianxiangMa/Compositional-NeRF \
[3] https://arxiv.org/pdf/2302.01579.pdf

