# SMS Spam Detector

A web application that tells you if a given SMS is spam or not spam. It uses a ML model trained on the [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and is built using Flask. Project made by [Subhashis Suara](https://www.subhashissuara.com/) & [Siddharth Das](https://www.linkedin.com/in/siddharth-das-2108at).

## About Dataset

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

## How does it work?
Initially we have dropped the unnecessary columns i.e. 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'. We have renamed columns i.e. v1 and v2 to label and message respectively. For classification later we replaced non spam to 0 and spam to 1. We first started to preprocess our data. All the special characters are removed from the string. We have also removed the stop words. We then used Porter Stemmer for stemming the words. After joining the stemmed words we have created a corpus of messages. The next step is to create Bag of Words model which we have done using CountVectorize function. We have then fitted the multinomial Naive Bayes to the training set and then saved the model in a file.

For deployment purpose we have used Flask. Using pickle library we load the Multinomial Naive Bayes model and CountVectorizer object. First the home page is displayed. The next step is to fill the sms(message) in the text field. After clicking the detect button , flask uses the Multinomial model to verify if the message is spam or not. The output is then displayed on the webpage.
