# EmotionAnalysis
The program aims to classify X posts (tweets) into six emotion categories (Sadness, Joy, Love, Anger, Fear, and Surprise) through supervised machine learning. In addition, the trained model and some other developed features are used in an in-the-wild study with new tweet data to generate a prototype of the an emotion analysis feature for discussions on X. 

## Research Paper
This project is developed for the research paper "Leveraging Emotion Detection to Enhance User Understandingon Social Media." The paper drafted was for the Pioneer Research Program and currently nominated for the _Pioneer Research Journal_. Read the full paper [here](https://drive.google.com/file/d/1N6pzBUV7BOJbk2UjgcS-xTU2GBswEUc9/view?usp=sharing).

### Abstract
Emotion plays a crucial role in human intelligence and interactions, and emotion detection technologies have been developed and applied in various contexts, such as for client feedback analysis, public opinion, and political research. However, the application of these technologies has been primarily in commercial and academic domains; there remains a significant gap to communicate detected emotions to the users. In today’s world where internet users consume large amounts of opinionated information and news from social media, increased awareness of the emotions regarding a subject of discussion can enhance user understanding and communication. Therefore, this paper proposes the integration of emotion detection technology into X (formerly Twitter) by developing an emotion analysis feature that displays a summary of the emotions expressed in reply to a post. The feature aims to expose users to a diverse array of emotions and leverage their curiosity to encourage exploration of opposing opinions to promote a more balanced and informed consumption of online content.

## Dataset 
The training dataset used was the [“Emotions” dataset](https://doi.org/10.34740/KAGGLE/) by Nidula Elgiriyewithana. It consists of 393,822 English X messages annotated with the six emotion categories. 

For the in-the-wild study, post replies were scraped from X. Data collection centered on two highly popular posts a political event: Joe Biden’s decision to withdraw from the 2024 presidential race. The data used included: 
- 191 replies to [Joe Biden’s post announcing his decision](https://x.com/JoeBiden/status/1815080881981190320)
- 182 replies to [CNN's subsequent announcement](https://x.com/CNN/status/1815085892987478441)

The following data were extracted from each reply:
- User identification
- Text content
- Number of likes
- Number of reposts
- URL to the reply
- URL to the user profile
- Date of posting

## Repository Outline 
### Data/
- tweets.csv: Emotions Dataset
- letter.csv: Biden's post replies
- news.csv: CNN's post replies

### Source/
- preprocess.py: Text preprocessing functions
- train.py: Model training script
- visualize.py: Learning curve visualization
- evaluate.py: Model evaluation with a series of metrics
- confidence_evaluation.py: Model decision confidence evaluation and visualization with decision scores
- confidence_metrics_by_emotion.py: Model decision confidence comparison of each class
- in_the_wild_study.py: Scraped data prediction with trained model + additional features
- wordclouds.py: Visual wordcloud generations on scraped data
  
## Dependencies
To install the required dependencies, run the following command:
```
pip install -r requirements.txt
```

## License 
This project is licensed under the MIT License. 
