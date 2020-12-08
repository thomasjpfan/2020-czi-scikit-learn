+++
title = "scikit-learn in 2020"
outputs = ["Reveal"]
+++

# scikit-learn in 2020

{{< grid middle >}}

{{< g 1 >}}
{{< figure src="images/czi_logo.png" width="300px">}}
{{< /g >}}

{{< g 1 >}}
{{< figure src="images/columbia.png" width="300px">}}
{{< /g >}}

{{< g 1 >}}
{{< figure src="images/scikit-learn-logo.png" width="300px">}}
{{< /g >}}


{{< /grid >}}


Thomas J. Fan

**Data Science Institute @ Columbia University**

<!-- {{< social >}} -->
{{< talk-link 2020-czi-scikit-learn >}}

---

# Contents 📖

1. Organizational Updates @ scikit-learn
2. Advances in scikit-learn
3. Improvements of the new fast gradient boosting models

---

{{% section %}}

# Organizational Updates @ scikit-learn
{{< figure src="images/scikit-learn-logo.png" width="300px">}}

---

{{< figure src="images/issues-and-prs.png">}}

---

# Triage Team!
{{< figure src="images/triage-team.png" height="500px">}}


---

# Triage Team Responsibilities

{{< figure src="images/labels.png" height="500px">}}

- Labels for issues and PRs
- Determine if a PR must be relabeled as stalled
- Close duplicate issues and issues that cannot be replicated

---

# Data Umbrella Sprint on 2020-06-20

{{% grid %}}


{{% g 2 %}}
- Organized by Reshama Shaikh of [Data Umbrella](https://www.dataumbrella.org) to increase the participation of underrepresented persons in data science
- Participants from ten different countries joined
- The attendees were evenly split by gender
- For more information, [see the sprint report](https://reshamas.github.io/data-umbrella-scikit-learn-online-sprint-report/)
{{% /g %}}

{{% g 1 %}}

{{< figure src="images/data-umberalla.gif" height="300px" >}}

{{% /g %}}

{{% /grid %}}

---

# New core developer: @lorentzenchr

- Lead Generalized Linear Models development:
    - **Tweedie distributions**
    - **Poisson**: Tweedie with `power=1, link='log'`
    - **Gamma**: Tweedie with `power=2, link='log'`

```python
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import GammaRegressor
```

{{< figure src="images/poisson_gamma_tweedie_distributions.png" width="1000px">}}

---

# Examples of distributions

- Risk modeling / insurance policy pricing:
    - number of claim events / policyholder per year (Poisson)
    - cost per event (Gamma)

- Agriculture / weather modeling:
    - number of rain events per year (Poisson)
    - amount of rainfall per event (Gamma)

---

### Usage Examples

- [User Guide](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression)
- [Poisson regression and non-normal loss](https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html#sphx-glr-auto-examples-linear-model-plot-poisson-regression-non-normal-loss-py)
- [Tweedie regression on insurance claims](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html#sphx-glr-auto-examples-linear-model-plot-tweedie-regression-insurance-claims-py)
{{< figure src="images/glm-dist.png" >}}

{{% /section %}}

---


{{% section %}}

# Advances in scikit-learn 🛠

---

# Showcasing Work by Nicolas Hug
## [@nicolashug](https://github.com/nicolashug)

---

# Sequential Feature Selection

Greedy procedure that iteratively finds the best new feature to add to the set of selected features.

```python
from sklearn.feature_selection import SequentialFeatureSelector
```

- [User Guide](https://scikit-learn.org/dev/modules/feature_selection.html#sequential-feature-selection)
- [Model-based and sequential feature selection](https://scikit-learn.org/dev/auto_examples/feature_selection/plot_select_from_model_diabetes.html#sphx-glr-auto-examples-feature-selection-plot-select-from-model-diabetes-py)

---

# Successive Halving

A state of the art method to explore the space of the parameters and identify their best combination.

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
```


---

# What is Successive Halving?


{{< figure src="images/halving.png" width="650px">}}

- [User Guide](https://scikit-learn.org/dev/modules/grid_search.html#successive-halving-user-guide)
- [Comparison between grid search and successive halving](https://scikit-learn.org/dev/auto_examples/model_selection/plot_successive_halving_heatmap.html#sphx-glr-auto-examples-model-selection-plot-successive-halving-heatmap-py)

---

# Major Improvements to Documentation

- [Random State](https://github.com/scikit-learn/scikit-learn/pull/18363)
- [Cross-Decomposition](https://github.com/scikit-learn/scikit-learn/pull/17095)
- [LDA and QDA](https://github.com/scikit-learn/scikit-learn/pull/16243)
- [Isotonic Regression](https://github.com/scikit-learn/scikit-learn/pull/16234)
- [Gradient Boosting](https://github.com/scikit-learn/scikit-learn/pull/16178)
- [Partial Dependence Curves](https://github.com/scikit-learn/scikit-learn/pull/16114)
- [Fast PDP for Trees and Random Forest](https://github.com/scikit-learn/scikit-learn/pull/15864)

---

# Estimator API + `check_estimator`

- Refactored `check_estimator` for better maintainability
- [Strict mode](https://github.com/scikit-learn/scikit-learn/pull/17361) to decide what is the minimal API of an `scikit-learn Estimator`
- [SLEP010](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html) `n_features_in_` + `_validate_data` method
    - Builds scaffolding to enable supporting for different array types

---

# HTML representation of HTML (default)

```python
clf
```

```python
Pipeline(steps=[('preprocessor',
                 ColumnTransformer(transformers=[('num',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(strategy='median')),
                                                                  ('scaler',
                                                                   StandardScaler())]),
                                                  ['age', 'fare']),
                                                 ('cat',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(fill_value='missing',
                                                                                 strategy='constant')),
                                                                  ('onehot',
                                                                   OneHotEncoder(handle_unknown='ignore'))]),
                                                  ['embarked', 'sex',
                                                   'pclass'])])),
                ('classifier', LogisticRegression())])
```

---

<section data-noprocess>
  <h1>HTML representation of HTML (now)</h1>
{{% markdown %}}
```python
from sklearn import set_config
set_config(display='diagram')
clf
```
{{% /markdown %}}
<style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="8a7af986-706b-472d-9c70-51cc6a9caba8" type="checkbox" ><label class="sk-toggleable__label" for="8a7af986-706b-472d-9c70-51cc6a9caba8">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('preprocessor',
                 ColumnTransformer(transformers=[('num',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(strategy='median')),
                                                                  ('scaler',
                                                                   StandardScaler())]),
                                                  ['age', 'fare']),
                                                 ('cat',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(fill_value='missing',
                                                                                 strategy='constant')),
                                                                  ('onehot',
                                                                   OneHotEncoder(handle_unknown='ignore'))]),
                                                  ['embarked', 'sex',
                                                   'pclass'])])),
                ('classifier', LogisticRegression())])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="d8df9348-923b-4289-852b-bc0c9f014af6" type="checkbox" ><label class="sk-toggleable__label" for="d8df9348-923b-4289-852b-bc0c9f014af6">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[('num',
                                 Pipeline(steps=[('imputer',
                                                  SimpleImputer(strategy='median')),
                                                 ('scaler', StandardScaler())]),
                                 ['age', 'fare']),
                                ('cat',
                                 Pipeline(steps=[('imputer',
                                                  SimpleImputer(fill_value='missing',
                                                                strategy='constant')),
                                                 ('onehot',
                                                  OneHotEncoder(handle_unknown='ignore'))]),
                                 ['embarked', 'sex', 'pclass'])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="cf0badb5-8104-45f6-9f41-caadbc17faeb" type="checkbox" ><label class="sk-toggleable__label" for="cf0badb5-8104-45f6-9f41-caadbc17faeb">num</label><div class="sk-toggleable__content"><pre>['age', 'fare']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="58dac894-1abe-4727-b163-891a83fdd07b" type="checkbox" ><label class="sk-toggleable__label" for="58dac894-1abe-4727-b163-891a83fdd07b">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy='median')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="28283f31-cfc1-455a-98cd-67b9ddd070d6" type="checkbox" ><label class="sk-toggleable__label" for="28283f31-cfc1-455a-98cd-67b9ddd070d6">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="3a74ea9a-e73c-4481-a770-6c0cf69209cb" type="checkbox" ><label class="sk-toggleable__label" for="3a74ea9a-e73c-4481-a770-6c0cf69209cb">cat</label><div class="sk-toggleable__content"><pre>['embarked', 'sex', 'pclass']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c62e0c4e-902b-43b2-8e14-ece22b60f599" type="checkbox" ><label class="sk-toggleable__label" for="c62e0c4e-902b-43b2-8e14-ece22b60f599">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(fill_value='missing', strategy='constant')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="35df2cb1-6742-4587-b270-e43745dbb550" type="checkbox" ><label class="sk-toggleable__label" for="35df2cb1-6742-4587-b270-e43745dbb550">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(handle_unknown='ignore')</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="dc06abd2-ea7e-4f28-af77-0fa90148eca3" type="checkbox" ><label class="sk-toggleable__label" for="dc06abd2-ea7e-4f28-af77-0fa90148eca3">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div>
</section>

{{% /section %}}

---

{{% section %}}

# Improvements of the new fast gradient boosting models 🌲

```python
from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
```

---

# Missing Value Support

```python
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

X = np.array([0, 1, 2, np.nan]).reshape(-1, 1)
y = [0, 0, 1, 1]

gbdt = HistGradientBoostingClassifier(min_samples_leaf=1).fit(X, y)
```

---

# Gradient Boosting with Poisson Distributions

```python
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

reg = HistGradientBoostingRegressor(loss="poisson")
```

{{< figure src="images/glm-dist.png" >}}

[Gradient Boosting Regression Trees for Poisson regression example](https://scikit-learn.org/dev/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html#gradient-boosting-regression-trees-for-poisson-regression)

---

# Categorical Support

```python
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

categorical_mask = ([True] * n_categorical_features +
                    [False] * n_numerical_features)

reg = HistGradientBoostingRegressor(categorical_features=categorical_mask)
```

---

# Categorical Advantages

{{< figure src="images/categorical-hist.png" height="500px">}}

[Categorical Feature Support in Gradient Boosting](https://scikit-learn.org/dev/auto_examples/ensemble/plot_gradient_boosting_categorical.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-categorical-py)


{{% /section %}}

---

# Conclusion

{{< figure src="images/scikit-learn-logo.png" width="200px">}}

**scikit-learn 0.24rc** has been released!

```bash
pip install scikit-learn==0.24.0rc
```

1. Organizational Updates @ scikit-learn
2. Advances in scikit-learn
3. Improvements of the new fast gradient boosting models

<p><br></p>

**Thomas J. Fan**
{{< social >}}

{{< talk-link 2020-czi-scikit-learn >}}


