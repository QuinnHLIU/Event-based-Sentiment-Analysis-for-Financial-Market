# Event-based-Sentiment-Analysis-for-Financial-Market

A canonical project estimating sentiment factor in Chinese financial market, based on machine learning techniques and textual data.


## Introduction

NLP method typically relies on suspicious initial manual labels. This introduces limits to sample size and may suffer from significant bias.

This project is meant to tackle issue through a proposed "event-labelling strategy".


## Methodological Approach

### Step 1. Scape Raw Data

From comment data on stock forum.

Code: **Scrape.py**


### Step 2. Define Risk Events

The most salient one is defult events of bond issuers among listed firms (approx. 10%). Data sourced from WIND and CHOICE. Code: **RiskEvent.do**.

> Actual defaults only happen since 2015 and the counts are rare. To extend sample size, an alternative news-based approach is also conducted.
> The events are then defined at sector level. Code: **GoogleNews.py**


### Step 3. Construct Event-based Labels

The comments immediately follow a risk event are thus identified as "NEGATIVE" comments. Word vectorization and Bag-of-words model are then conducted
following standard procedure.

Code: **WordVec.py**


### Step 4. Model Training and Performance

The labeled data is then fed into CNN and LSTM models for structural estimation. Performance of the trained model is then tested on 12w+ sample data.

Code: **CNN.py**, **LSTM.py**


### Step 5. Asset Pricing Implication

Formalize the sentiment factor as a deviation from frictionless pricing kernel. Then conduct a horse race with standard factors and show the identified
sentiment factor is useful

$$

E_t[R_t] = r-\lambda +\eta\sigma_x\rho_x +S_t\frac{v'[S_t]}{v[S_t]}\frac{\rho}{1-\rho}(\rho_x\sigma_x-\rho_w\sigma_w)

$$




