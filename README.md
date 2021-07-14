## Customer Churn Model
This repository creates a REST API, written in Flask, that predicts the probability a customer will churn, 
that is, cancel their subscription. This probability is generated from a machine learning model, also
trained in this repo. In sum, the repository contains all the code necessary to build a churn prediction
system. 

Start by cloning the repository. <br>
$ git clone git@github.com:micahmelling/applied_data_science.git

## Interacting with the Flask API
Spin up a local server to test the Flask application by adding the following boilerplate. <br>
if __name__ == "__main__":
&nbsp;&nbsp;&nbsp;app.run(debug=True)

You can then run a payload through the following endpoint: http://127.0.0.1:5000/predict.

Sample input payload: <br>
```javascript
{
    "activity_score": 0.730119,
    "propensity_score": 0.180766,
    "profile_score_new": 22,
    "completeness_score": 41.327,
    "xp_points": 18.2108,
    "profile_score": 20.0989,
    "portfolio_score": 25.1467,
    "mouse_movement": 0.5,
    "average_stars": 1.5,
    "ad_target_group": "level_4",
    "marketing_message": "level_3",
    "device_type": "level_3",
    "all_star_group": "level_2",
    "mouse_x": "level_5",
    "coupon_code": "level_10",
    "ad_engagement_group": "level_2",
    "user_group": "level_3",
    "browser_type": "level_2",
    "email_code": "level_1",
    "marketing_creative": "level_4",
    "secondary_user_group": "level_11",
    "promotion_category": "level_9",
    "marketing_campaign": "level_8",
    "mouse_y": "level_8",
    "marketing_channel": "level_16",
    "marketing_creative_sub": "level_1",
    "site_level": "level_12",
    "acquired_date": "2015-06-09",
    "client_id": "1963820"
    }
```

The model will return a payload like the following. <br>

```javascript
{
'prediction': 0.14, 
'high_risk': 'no', 
'response_time': 0.471, 
'ltv': 0
}
```

The repo also comes with a Dockerfile that can be used to create a Docker image. <br>
$ docker build -t churn . <br>
$ docker run --rm -it churn

The docker run command will spin up http://127.0.0.1:8000, and you can now hit the predict endpoint 
once again. Please note the port is different compared to using the default Flask server. 

The app_settings.py file in root gives us the ability to update the models our API uses along with the option
to update key global variables. These values represent ones we don't intend to change frequently or want 
to run through and be tested in our CI/CD pipeline. 

Our application also responds to a configuration. The configuration allows us to update straightforward 
values we might want to change somewhat frequently. For example, we might want to almost effortlessly update
the percentage of requests that receives a holdout treatment. Our config values are housed in MySQL in 
churn_model.prod_config or churn_model.stage_config, depending on the environment, A config UI is available
that allows us to easily add and update config values via the config-refresh endpoint in our API. 

## How to the Use Repo - Model Training
The modeling directory houses code for training models that predict churn. Kicking off train.py will train 
a new set of models, defined in config.py. Within the modeling directory, a subdirectory is created for 
each model run, all named with a model UID. This allows us to version models and keep records of every run. 
Each modelsubdirectory is uploaded to S3, so we can clear local versions once our model directory becomes 
cluttered.

Below is a rundown of all the files we might wish to adjust in the modeling directory. <br>
<li>
config.py - holds global variables and configurations for training models. MODEL_TRAINING_DICT declares all 
the types of models we want to train. MODEL_EVALUATION_LIST details all the metrics we want to use on our 
test set.  
</li>
<li>
pipeline.py - holds scikit-learn pipelines to be tuned via cross validation and serialized with joblib. 
</li>
<li>
model.py - houses functions for training and tuning models. 
</li>
<li>
evaluate.py - stores functions for evaluating our models. 
</li>
<li>
explain.py - holds functions for explaining models on both a global and local scope. 
</li>
<li>
train.py - brings together functions from the foregoing files to actually train, evaluate, and explain a 
series of models. 
</li>

## Other Key Repo Files
A number of other files exist in the repo. Below are the most important ones. <br>
<li>
data/db.py - houses functions to query databases and other data stores.
</li>
<li>
helpers/app_helpers.py - stores functions to aid in running app.py.
</li>
<li>
helpers/model_helpers.py - holds functions to aid in running modeling/train.py.
</li>
<li>
tests/test_app.py - houses unit tests for app.py.
</li>

## Production Releases
A production release of new code or models can be accomplished by pushing to the main branch of the 
remote repository. This will kick off a CI/CD pipeline build that will test the code changes, release them
to staging, and then release them to production upon manual approval. 