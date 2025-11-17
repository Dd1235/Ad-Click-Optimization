# Ads Click Optimization Project Explanations

Let's go over some theory and terms. And we will solve some optimization problems, albeit not multi objective! That would be a great extension to this project.

The click through rate of an ad is the probability that a user clicks on the ad when it is shown. In this project, we aim to optimize the allocation of a fixed budget across various ad impressions to maximize the expected number of clicks. Sounds gibberish, will go over the details.

If your ad was shown 10,000 times and got 150 clicks then the click through rate is 150/10,000 = 0.015 = 1.5%.

Inventory is all the places an ad can be shown. Each impression is a single opportunity to show an ad to a user. So one slot where an impression has chance to occur.

Suppose you want to spend Rs.1000 on ads. You have inventory choices.

Inventory (i) | Example| Cost per impression| CTR
i = 1| Instagram Story| ₹1.0 |2.5%
i = 2| YouTube pre-roll| ₹1.5| 4.0%
i = 3 |Google Search |₹2.0 |10.0%

Cost per impression is, hey I want to show my ad on a Google search, how much will it cost me? 2 rupees. CTR is, what is the probability a user will click on it once it is shown to them? 10%.

So buying an impression on google search is twice as expensive than say instagram story, but the CTR is 4 times higher. So you might want to allocate more budget to google search.

Now for an ad, you don't really know the click through rate. So you build a model that can predict CTR. You use ML techniques for binary classification, take a bunch of features about your ad, user, context, time of day, device type etc and build a model that can predict CTR.
You can use this predicted CTR to optimize on all your constraints.

So how is a cost per impression calculated? It is what the advertiser pays when their ad wins the auction for that impression. This is NOT fixed!

Every ad impression (e.g., Instagram story, Google search top result, YouTube mid-roll) is sold through a real-time bidding auction:

    User opens app / page
    The platform triggers an auction for that one impression
    Multiple advertisers bid how much they’re willing to pay
    The highest bidder wins
    The winner pays the price (usually second-price or VCG variant)
    That price = cost for that specific impression.

So cost per impression is dynamic, it depends on the auction. Essentially you need to have a bidding strategy that can bid optimally for each impression. This is a different probleem altogether! But this tells you why the cost per impression varies every single time.

Platforms usually show you CPM(cost per mille) - cost per 1000 impressions. So if CPM is ₹50, cost per impression is ₹0.05.

But behind the scenes:

    - CPM is aggregated over many auctions
    - Each individual impression has its own price
    - CPM = (sum of impression prices) / (1000)

So there aren't any datasets with "cost per impression" - it fluctuates real time.

Google or Meta use second-price auction variants.

Example: - Advertiser A bids ₹10 - Advertiser B bids ₹8 - Advertiser C bids ₹5 - A wins - A pays ~₹8.01 (second price + epsilon)

Twitter/X uses first-price auctions, pay your bid.

You get the idea.

I ran some experiments on the Azavu dataset (one of the biggest ads datasets out there, 40M rows, a few dozen cols, don't exactly remember), but its from 2014. [will insert link to it later]

I will actually use the Criteo Attribution Modeling for Bidding Dataset from 2017, some 16.5 million rows. Criteo 1TB click logs dataset from 2023 is the biggest though, for our demonstration purposes it doesn't matter too much because we will be using a small sample of the data that fit our compute limitations.

Refer this [paper](https://arxiv.org/pdf/2502.12103) on the Criteo datasets.
Refer

- [their site](https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/)

Below is a cleaned-up description of the dataset (originally from the Criteo site). The data are subsampled and anonymized.

This dataset is a sample of 30 days of Criteo live traffic. Each row represents a single ad impression and contains information about:

- the impression context (anonymized categorical features),
- whether the impression was clicked,
- whether a conversion happened within 30 days, and
- auction-related costs (transformed values).

Fields (tab-separated):

- `timestamp` — timestamp of the impression (starts at 0). The dataset is sorted by this field.
- `uid` — unique user identifier.
- `campaign` — unique campaign identifier.
- `conversion` — 1 if a conversion occurred within 30 days after the impression, 0 otherwise.
- `conversion_timestamp` — timestamp of the conversion, or `-1` if none.
- `conversion_id` — unique identifier for each conversion (or `-1` if none); useful to reconstruct timelines.
- `attribution` — 1 if the conversion was attributed to Criteo, 0 otherwise.
- `click` — 1 if the impression was clicked, 0 otherwise.
- `click_pos` — position of the click before conversion (0 = first-click).
- `click_nb` — number of clicks prior to conversion (can be >1).
- `cost` — price paid by Criteo for this display (transformed; not the real raw price).
- `cpo` — cost-per-order for attributed conversions (transformed; not the raw price).
- `time_since_last_click` — time since the previous click for the given impression (seconds).
- `cat1`…`cat9` — nine anonymized contextual categorical features. In experiments these were mapped to fixed-size vectors using the hashing trick.

Note: all monetary values are transformed versions provided by the dataset provider and should be treated as relative indicators rather than exact prices.

Some observations:

Now I am not sure if uid is needed, say I used it for prediction, it's possible it requires one hot encoding to not suffer from drawing correlation between say the number assigned to user and the target variable. But one hot encoding for millions of users is not feasible. Also uid does not describe the user in any way, how will it work for new users?

Campaign identifier, I assume is needed, because different campaigns will have different CTRs. Conversion_id might be dropped, I don't need it for reconstruction of timeline.

I am not sure how to deal with the attribution. If criteo was attributed, perhaps I can drop those rows? Need to do EDA first.

Cost is transformed, so perhaps it doesn't reflect real cost in dollars (I assume? anyway units won't matter since its transformed). Assuming the linear relationships would be preserved and it still holds good meaning. This is better that generating a random cost value like I did in Azavu dataset experiment.

What is this cpo column?

YOu're criteo.

An advertiser says:
“I will pay you ₹300 for each user who buys. Show ads however you want.”

This is a CPA contract.
Now suppose an impression led to a conversion, now the guy who is advertising is happy, that he sold a product using this ad, so he will pay you extra, on top of whatever was the cost for buying that impression.

That cpo.

Btw criteo's job is to buy ad impressions from other companies. So Say criteo decides to buy an impression from google, they have to pay the cost, that is money to host that ad on that impression. But they get money back when the actual guy who is selling the product pays them for the conversion.

cat1,...,cat9 describe things like user segment, publisher cateogry, etc. but are anonymized.

This dataset includes ONLY ads Criteo purchased for its clients.
There are no Google Ads, no Facebook ads, no publisher-owned ads, no organic events.
Everything is from Criteo’s perspective as a Demand-Side Platform (DSP).

But here is the confusing part, each impression is unique, cannot group them to get a general CTR. We are already given if an impression was clicked or not.

In this dataset:
Each impression is one auction outcome
So each row is effectively its own inventory item
There is no separate grouping.
This is EXACTLY how real-time bidding works:
You don’t buy “slots”
You buy individual impression opportunities
So modeling CTR per row is natural.

So for each impression based on features, lets calculate a pCTR -> probability of click based on the features, ofcourse we already know if it was clicked or not, but this is more useful to learn as we get a new opportunity with certain features, and we want to have a prediction of click, and do allocation based on that.

The `Experiments.ipynb` file is provided by criteo, credits to them. Just downloaded it for my reference.

## Optimization problem 1 summary

- Worked with 50,000 impressions spread across 637 campaigns. After scaling the transformed costs, the total spend in this slice is ₹1.30 million equivalents, and we only allow ourselves to redeploy 10% of it (₹130,153) as the budget `B`.
- Each impression decision is a variable `x_i` between 0 and 1. Zero means skip that impression, one means we fully buy it, and a fractional value says “if this impression type reappears, spend that fraction of the time.” This fractional relaxation mirrors the standard knapsack trick and keeps the problem convex and easy to solve.
- A convex program that maximizes total predicted clicks `pCTR @ x` under the budget constraint allocates the entire ₹130k and returns an expected 11,277 clicks. A greedy fractional-knapsack rule that sorts impressions by pCTR-per-cost and fills the budget sequentially reaches 11,277 clicks as well, with a gap of only 6e-06 clicks, so the greedy intuition is perfectly aligned with the convex optimum for this linear objective.
- Grouping the solution by campaign shows how skewed the inventory is. Campaign 7 alone contributes 3,177 impressions and 6.4% of historical cost, so even the unconstrained optimizer sends ₹9.4k (7.2% of the budget) its way, while hundreds of smaller campaigns receive pennies. This table is a practical way to explain the model to a business partner (“top 20 campaigns eat most of the spend; the tail gets almost nothing”).
- To add simple fairness we give every campaign a minimum budget equal to 5% of its historical cost share and cap every campaign to at most 150% of its share with a hard upper limit of 8% of the total budget. With these constraints the solver still uses the entire ₹130k but the objective drops to 10,626 clicks, so we sacrifice about 651 clicks (~6%) to keep everyone in-bounds.
- The fairness visualization confirms the story: campaign 7 is still the best performer but it is now clamped exactly at the 8% ceiling (₹10.4k). More than 300 campaigns hit their max cap, so the policy mostly throttles the densest campaigns, while 16 tiny campaigns sit on their minimum budgets to maintain exposure. This is the expected outcome when the dataset has a heavy head-and-tail distribution.
