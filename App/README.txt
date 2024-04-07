Text to Song Recommendation

To allow users to create more meaningful content, our project aims to recommend users audio that would best fit their post. The app
uses text-to-sentiment models to predict the emotion of the user's captions before recommending audio with matching emotions predicted
by a music-to-emotion Machine learning model.

Requirements:

pip install streamlit
pip install numpy
pip install protobuf-3.20.0
pip install tensorflow
pip install tf-keras

Running the app:
1. in your terminal run the command "pip install streamlit"
1. Open the file directory in your terminal
2. run the command "streamlit run app.py"
3. Your local browser should open, please do wait for the webpage to finish loading as it may take a while.

Using the app:
1. Enter your caption
2. Press enter to search
3. Get your music recommendations

Credits:
Credit to bhadresh-savani for providing open sourced distilbert base uncased emotion text-to-emotion classifier model that we used
for caption to emotion prediction.