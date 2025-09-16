## AI Eng 

A space to try out the latest tooling available for AI Engineering. Not the space for side projects.

_I can't promise the most beautiful code_

### 📁  Directory 

-----

#### 🔧 [Language Detection App](C:\Users\jorda\PycharmProjects\AI_Eng\Language Detection App)

Model trained to detect the language being passed (NaiveBayes). The model is stored as a pickel file but hosted via docker on Heroku.  

Notes on the oauth step:
Step 1: Client requests access - the dependencies indicated in the endpoint specify the user must have a valid token to continue. 
Step 2: Client logs in through the /token endpoint - providing the correct credentials means that a token is returned every call.  
Step 3: Upon receiving the token, the client stores it local storage/memory/cookies
Step 4: Now every time the client attempts to access a protected endpoint it has a temporary pass to do so. 