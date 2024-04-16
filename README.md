# Image Classification

## Overview
This learning project is an image classifier built with the fastai library. 
A local server is made using flask and then created a simple wepage to run the model (used javascript and HTML/CSS)
It categorizes images into the following classes:

- **Art**: Drawings, anime, and paintings
- **Meme**: Funny cartoons or memes
- **Silly Image**: Photoshopped or "fake" images
- **Pets**: Cats and dogs
- **Selfie**: Selfies
- **Sexual**: Images with sexual content
- **Text-based**: Images predominantly containing text
- **Generic**: All other images that don't fall into the above categories

### Implementation Details
The classifier has been developed with fastai, incorporating standard procedures such as data processing, preprocessing, and hyperparameter tuning. The approach and methods are primarily based on the textbook "Deep Learning for Coders with fastai & PyTorch."

### Data Collection
The images were collected from Reddit using Beautiful Soup. A quick manual review was conducted to remove any mistakes and outliers from the dataset.

#### Reference
The textbook referenced for this project is:
"Deep Learning for Coders with fastai & PyTorch" by Jeremy Howard and Sylvain Gugger.


