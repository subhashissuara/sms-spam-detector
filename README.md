# SMS Spam Detector

A web application that tells you if a given SMS is spam or not spam. It uses a ML model trained on the [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and is built using Flask. Project made by [Subhashis Suara](https://www.subhashissuara.com/) & [Siddharth Das](https://www.linkedin.com/in/siddharth-das-2108at).

## About Dataset

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged according being ham (legitimate) or spam.

## How does it work?

From the dataset, we dropped unnecessary columns i.e. 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'. Then we renamed the columns i.e. v1 and v2 to label and message respectively. For classification purpose we replaced not spam label with 0 and spam with 1. We preprocessed our data by removing all special characters and stopwords. We then used Porter Stemmer for stemming the words. After joining the stemmed words we have created a corpus of messages. The next step was to create a Bag of Words model which we accomplished using CountVectorize function from sklearn library. We then fit the Multinomial Naive Bayes to the training set and then saved the model using the pickle module.

For deployment purpose we have used Flask. Using the pickle module, we loaded the Multinomial Naive Bayes model and the CountVectorizer object. The Flask app has 2 routes. First route is the home page route and second route is the detect route. The home page consists of a text fiels to input the sms message and a button to detect the sms type. Upon clicking the detect button, the Flask app uses the Multinomial model to classify if the message is spam or not spam. The detect page then displays the appropriate output on the webpage.

## How to run the project?

Ensure that you have Python installed and enter the following commands in your terminal:

```
git clone https://github.com/subhashissuara/sms-spam-detector
pip install -r requirements.txt
python app.py
```
