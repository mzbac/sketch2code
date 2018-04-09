# Sketch2code

![Preview](https://github.com/ashnkumar/sketch-code/blob/master/header_image.png)

## Generating HTML Code from a hand-drawn wireframe
a simple deep learning model that takes hand-drawn web mockups and converts them into working HTML code. It uses an [image captioning](https://towardsdatascience.com/image-captioning-in-deep-learning-9cd23fb4d8d2) architecture to generate its HTML markup from hand-drawn website wireframes.

Heavily inspired by Ashwin Kumar's blog post, For more information, check out his post: [Automating front-end development with deep learning](https://blog.insightdatascience.com/automated-front-end-development-using-deep-learning-3169dd086e82)

## Model architecture
![model](https://raw.githubusercontent.com/mzbac/sketch2code/master/model_architecture.png)
## Download the data
```bash
bash get_data.sh
```
## Docker demo settings
```bash 
docker pull mzbac/sketch2code
docker run -p 5000:5000 mzbac/sketch2code
```
browser to localhost:5000

## Load pre-trained weights
```python
encoder = torch.load('model_weights/encoder_resnet34_0.061650436371564865.pt')
decoder = torch.load('model_weights/decoder_resnet34_0.061650436371564865.pt')
```
## Pre-trained weight preview:

![loss_0.061](https://raw.githubusercontent.com/mzbac/sketch2code/master/image_sketch2code_loss_0.061.png)

## Pre-trained model's BLEU score 
- 0.974
