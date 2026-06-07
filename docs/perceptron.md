# Perceptron: Teaching a Machine to See Light

Most machine learning tutorials start with definitions or mathematics. We will start with a problem and work toward a solution. Along the way, we will discover the fundamental ideas behind machine learning and eventually arrive at the Perceptron.

## How can a computer tell whether a room is bright enough?

Imagine a room with ten light bulbs, each either `ON (1)` or `OFF (0)`. We want a computer to answer one simple question:

> Is this room bright enough?

A straightforward approach:

```python
def is_bright(lights):
    threshold = 5
    return sum(lights) > threshold
```

This works. But where did the number `5` come from?

We guessed. Or rather, we reasoned: "half of 10 is 5, so more than 5 means bright." Either way, *we* supplied the answer. The computer just followed our instruction.

That's fine when the rule is obvious. But what if it wasn't? What if the threshold was 3, or 7, or 4.5? What if we simply didn't know?

This is the core problem that machine learning solves:

**What if we only have examples, and need the program to find the rule itself?**

## What does "learning from examples" actually mean?

Let's say we have a collection of observations — each one records the ON/OFF state of all ten bulbs and whether the room was considered bright:

```python
training_data = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # 0 lights on → dark
    ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 0),  # 5 lights on → dark
    ([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 0),  # 5 lights on → dark
    ([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], 1),  # 6 lights on → bright
    ([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 1),  # 7 lights on → bright
    ([0, 1, 1, 1, 1, 1, 1, 0, 0, 0], 1),  # 6 lights on → bright
]
```

Looking at these examples, most people quickly notice that rooms with five or fewer lights are dark, while rooms with six or more are bright. Your brain inferred the boundary from examples.

Can a computer do the same?

## Can we build something that learns the threshold instead of being told it?

Before thinking about *how* to learn, let's sketch the shape of what we want to build. We need something that:

- Holds a threshold as a value it can change over time
- Can make a prediction given a set of lights
- Can improve its threshold by looking at training examples

```python
class BrightnessDetector:
    def __init__(self):
        self.threshold = ???  # We don't know this yet

    def predict(self, lights):
        return 1 if sum(lights) > self.threshold else 0

    def train(self, training_data):
        pass  # We don't know how yet
```

The prediction logic is the same as `is_bright` — that part is unchanged. What's new is that `threshold` is no longer hardcoded.

The open question is: what should the threshold start as, and how should `train` change it?

## What should the threshold start as?

We don't know the right threshold — that's the whole point. So let's start with a random guess.

```python
def __init__(self):
    self.threshold = random.uniform(-20, 20)
```

A bad guess is actually useful. It lets us observe whether the learning process can recover from a wrong start — which is more convincing than getting lucky with a good initial value.

## How should the threshold change? Let's reason from a mistake.

Suppose the random starting guess was `9`.

For the case `6 lights on → bright`, the model predicts dark. That's wrong. The sum of lights (6) failed to clear the threshold (9). The threshold is too high, so we should lower it.

Now suppose the threshold starts at `2`. For the case `5 lights on → dark`, the model predicts bright. The threshold is too low, so we should raise it.

The mistake itself tells us what to do. There are only two kinds of mistakes:

| What happened              | Meaning            | Action     |
|---------------------------|--------------------|------------|
| Predicted dark, was bright | Threshold too high | Lower it   |
| Predicted bright, was dark | Threshold too low  | Raise it   |

No mistake? Don't touch the threshold — it already produced the right answer.

This is a remarkable property: **we don't need to know the correct threshold in advance**. Each wrong prediction carries enough information to tell us which direction to move. The data teaches the model, one correction at a time.

## But one correction isn't enough — we need to repeat this.

A single pass through the training data will fix some mistakes but probably not all. After adjusting the threshold for one example, the next example might require a different adjustment. And adjustments can interfere with each other.

The solution is to go through all the training examples repeatedly — making small corrections each time — until the threshold settles at a value that gets every example right.

Each full pass through the training data is called an **epoch**. After enough epochs, if a correct threshold exists, the corrections will stop — because the model will stop making mistakes.

This is the algorithm:

1. Start with a random threshold.
2. Go through every training example.
3. If the prediction is wrong, nudge the threshold in the right direction.
4. Repeat from step 2 until there are no more mistakes.

```python
import random

class BrightnessDetector():
    def __init__(self):
        self.threshold = random.uniform(-20, 20)

    def predict(self, lights):
        return 1 if sum(lights) > self.threshold else 0

    def train(self, training_data, epochs=10):
        for _ in range(epochs):
            for lights, label in training_data:
                prediction = self.predict(lights)
                if prediction != label:
                    if prediction == 0:
                        self.threshold -= 1
                    else:
                        self.threshold += 1
```

## From counting lights to measuring brightness

So far, every bulb has counted equally toward the total. But the real world rarely works that way.

A bedside lamp and a floodlight both count as one bulb, yet they produce very different amounts of light. Suppose our ten bulbs have known brightness levels:

```python
[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
```

If only the first and third bulbs are on:

```python
lights = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
```

the total brightness is `1.0 + 3.0 = 4.0`, not a raw count of 2. Instead of counting bulbs, we sum their brightness contributions.

But here's the twist: what if we don't know the brightness levels either? Just as we didn't know the correct threshold, we now don't know the importance of each bulb.

So we start with random guesses — one brightness level per bulb, plus the threshold. We initialize them all as random values between `-1` and `1`:

```python
class BrightnessDetectorWithWeights():
    def __init__(self):
        self.brightness_of_lights = [random.uniform(-1, 1) for _ in range(10)]
        self.threshold = random.uniform(-1, 1)

    def predict(self, lights):
        weighted_sum = self._weighted_sum(lights)
        return self._activation_function(weighted_sum)

    def _activation_function(self, z):
        """Step function: returns 1 if input > 0, else 0"""
        return 1 if z > 0 else 0

    def _weighted_sum(self, lights):
        """Calculate the weighted sum of the lights including threshold"""
        return sum(b * l for b, l in zip(self.brightness_of_lights, lights)) + self.threshold
```

Notice the threshold's role has shifted: rather than comparing the sum against it, we add it to the sum and compare the result against zero. Mathematically equivalent, but it sets us up for a cleaner update rule.

We call this combined value `activation`. Think of it as the perceptron's internal signal strength — how strongly it "feels" the input. A positive activation pushes the output toward 1 (bright); a negative activation pushes toward 0 (dark). The name comes from the idea of a neuron "activating": the higher the activation, the more likely the perceptron is to "fire" and output 1.

In real neural networks, this "firing" decision is made by an **activation function**. We've separated this into its own method `_activation_function`. It takes the activation value and applies the *step function*: output 1 if the activation is positive, 0 otherwise. This mirrors how biological neurons either fire or don't fire based on whether their input signal crosses a threshold.

## How do we train the brightness levels?

The same logic that guided threshold updates now applies to every brightness level.

Suppose `lights = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]` and the correct label is `bright (1)`, but the model predicts `dark (0)`.

The activation was too low. We need to raise it. Which brightness levels should we increase? Only the ones whose bulbs were actually ON — the bulbs that were OFF contributed nothing to the activation regardless of their brightness levels, so changing them would have no effect.

The reverse applies when we predicted bright but the room was dark: the activation was too high, so we nudge every ON bulb's brightness level downward.

We control the size of each nudge with a parameter called the **learning rate**:

$$\text{new_brightness} = \text{old_brightness} + \eta \times (\text{label} - \text{prediction}) \times \text{bulb_state}$$

Here, $(\text{label} - \text{prediction})$ is $+1$ when we predicted too low, $-1$ when we predicted too high, and $0$ when correct. The bulb state ($1$ or $0$) naturally zeroes out updates for OFF bulbs. The threshold follows the same rule with an implicit input of $1$:

$$b \leftarrow b + \eta \times (y - \hat{y})$$

## Why do we need a learning rate?

Imagine changing brightness levels by `100` every time a mistake occurs. The model would overshoot and bounce around wildly, never settling.

Instead, we make small corrections:

```python
learning_rate = 0.1
```

This means: move only one tenth of a unit in the required direction. Large learning rates learn quickly but risk overshooting. Small learning rates are more stable but take longer. `0.1` is a sensible starting point.

## The full training algorithm

```python
class BrightnessDetectorWithWeights():
    def __init__(self):
        self.brightness_of_lights = [random.uniform(-1, 1) for _ in range(10)]
        self.threshold = random.uniform(-1, 1)

    def predict(self, lights):
        weighted_sum = self._weighted_sum(lights)
        return self._activation_function(weighted_sum)

    def train(self, training_data, epochs=10, learning_rate=0.1):
        for _ in range(epochs):
            for lights, label in training_data:
                prediction = self.predict(lights)
                error = label - prediction  # +1, -1, or 0
                for i in range(len(lights)):
                    self.brightness_of_lights[i] += learning_rate * error * lights[i]
                self.threshold += learning_rate * error

    def _activation_function(self, z):
        """Step function: returns 1 if input > 0, else 0"""
        return 1 if z > 0 else 0
    
    def _weighted_sum(self, lights):
        """Calculate the weighted sum of the lights including threshold"""
        return sum(b * l for b, l in zip(self.brightness_of_lights, lights)) + self.threshold
```

If `error` is zero — correct prediction — nothing changes. Otherwise every brightness level whose bulb was ON gets nudged in the right direction, and so does the threshold. Repeat across enough epochs and everything converges to values that classify every example correctly.

## Why does starting between -1 and 1 still work?

Consider the output after training:

```
Initial Threshold: 0.55
Initial Brightness: [0.90, -0.61, -0.46, -0.79, -0.18, 0.46, 0.68, 0.81, -0.86, 0.48]
Accuracy before training: 0.39

Threshold after training: -4.55
Brightness after training: [0.40, 0.89, 1.34, 0.41, 0.92, 1.16, 0.48, 0.91, 1.34, 0.38]
Accuracy after training: 1.00
```

Notice that the learned values don't match the actual brightness levels we set up earlier:

```
Actual:  1.0, 2.0, 3.0, ...
Learned: 0.40, 0.89, 1.34, ...
```

That's fine — because the perceptron is not trying to discover the physical brightness of each bulb. It's trying to discover *a boundary that makes correct predictions*. Many sets of values can produce the same boundary. The learned values are just one valid solution.

What matters about the initialization is that the values are not all identical (otherwise every bulb would be updated by the same amount and the model could never distinguish between them) and not so large that early predictions are completely off course. Starting near zero satisfies both conditions. From there, the learning rule steers the values wherever they need to go.

## From room to formula: the Perceptron

We built our brightness detector from first principles. Now let's step back and name what we actually built.

A **perceptron** takes a list of numerical inputs, multiplies each by a learned **weight**, sums the results, adds a learned **bias** (our threshold), and outputs a binary decision based on the sign of that sum.

The terminology maps directly onto our room:

| Our concept                     | General name         |
|--------------------------------|----------------------|
| Bulb ON/OFF state               | Input ($x_i$)        |
| Brightness level of bulb i      | Weight ($w_i$)       |
| Threshold (added to sum)        | Bias ($b$)           |
| Combined brightness + threshold | Activation ($z$)     |
| Bright / Dark prediction        | Output ($\hat{y}$)   |

## Why is it called "bias"?

In our room analogy, we thought of this value as a *threshold* — a level the total brightness must exceed. But in the perceptron equation, it's added to the weighted sum rather than compared against it. This shifted perspective reveals why it's called **bias** in neural network terminology.

Think of the bias as controlling how easily the perceptron "fires." A positive bias makes the perceptron more inclined to output 1 (bright), even with weak inputs. A negative bias makes it more conservative, requiring stronger positive evidence to fire. It's like a *bias* in the perceptron's default behavior — a predisposition toward one output or the other before seeing any input.

Mathematically, you can view the bias as the weight of an extra input that is always 1:

$$z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b \cdot 1$$

This is why the bias updates just like any other weight, but with a constant input of 1.

Written as an equation, the activation is:

$$z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$$

Or more compactly:

$$z = \left(\sum_{i=1}^{n} w_i x_i\right) + b$$

The output is then:

$$\hat{y} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}$$

The learning rule generalizes the same way. For each training example with inputs $\mathbf{x}$, label $y$, and prediction $\hat{y}$:

$$w_i \leftarrow w_i + \eta\,(y - \hat{y})\,x_i \qquad \text{for each } i$$

$$b \leftarrow b + \eta\,(y - \hat{y})$$

where $\eta$ is the learning rate. When the prediction is correct, $y - \hat{y} = 0$ and nothing changes. When it's wrong, every weight is nudged in proportion to its input, and the bias (our threshold) is nudged unconditionally.

The bulbs and brightness levels were just a way in. The perceptron works the same way for any inputs — pixel values, temperatures, word counts, sensor readings — as long as the problem has the right shape. What shape? That's what the next section is about.

## Linear separability: what problems can a perceptron solve?

To understand what a perceptron can and can't do, we need to look at what the activation is really computing.

### The activation is a line

Start with the simplest case: one input. The activation becomes:

$$z = wx + b$$

Compare this to the equation of a straight line:

$$y = ax + b$$

They are identical in structure. $w$ plays the role of slope $a$, and $b$ is the intercept. The perceptron outputs `1` when $z > 0$, which means it fires when:

$$wx + b > 0 \quad\Longleftrightarrow\quad x > -\frac{b}{w}$$

The decision boundary is a single point on the number line. Everything on one side is class 0, everything on the other is class 1.

With two inputs:

$$z = w_1 x_1 + w_2 x_2 + b$$

Setting $z = 0$ gives a **line** in two dimensions. With three inputs, it's a **plane**. In general, the decision boundary is always a **hyperplane** — a flat, straight cut through the input space.

### What this means for classification

A dataset is **linearly separable** if a straight line (or hyperplane) exists that perfectly divides the two classes — all the 1s on one side, all the 0s on the other.

Our brightness problem was linearly separable. A single boundary at "more than 5.5 bulbs on" separated dark from bright without error.

For example, consider a simpler version with just one brightness value per room:

| Brightness | Label |
|-----------|-------|
| 1          | 0     |
| 2          | 0     |
| 3          | 0     |
| 7          | 1     |
| 8          | 1     |
| 9          | 1     |

A boundary at 5 separates them perfectly:

```
0  0  0  |  1  1  1
```

A perceptron can learn this boundary.

The **Perceptron Convergence Theorem** guarantees that if a linearly separable boundary exists, the learning algorithm will find it in a finite number of steps — regardless of where the values start. This is the theoretical underpinning of what we observed empirically: starting from random brightness levels, the model always converged to 100% accuracy.

### The limitation

Not every problem is linearly separable. Consider the XOR function:

| $x_1$ | $x_2$ | Output |
|-------|-------|--------|
| 0     | 0     | 0      |
| 0     | 1     | 1      |
| 1     | 0     | 1      |
| 1     | 1     | 0      |

Plot these four points and try to draw a straight line that puts the 1s on one side and the 0s on the other. It's impossible — the two classes are diagonally interleaved and no flat boundary can separate them. A perceptron will never converge on XOR.

A single perceptron can only carve the world in half with a straight cut. Problems that require curved or more complex boundaries are beyond its reach.

This limitation led researchers to connect many perceptrons together into layers, where each layer can carve the space into increasingly complex shapes. That is the foundation of modern neural networks.

The perceptron is therefore both a useful algorithm and the starting point upon which deep learning was built.

---

## Glossary: Key Concepts

Here is a quick reference for the terminology we've developed:

**Perceptron**  
A computational unit that takes multiple inputs, applies weights to each, sums them, adds a bias, and produces a binary output based on whether the result exceeds zero. The simplest form of an artificial neuron.

**Input ($x_i$)**  
A value (typically 0 or 1) fed into the perceptron, representing a feature of the data being classified.

**Weight ($w_i$)**  
A learned parameter that scales an input's contribution to the weighted sum. Determines how strongly each input influences the output.

**Bias ($b$)**  
A learned parameter added to the weighted sum, controlling how easily the perceptron outputs 1. A positive bias makes the perceptron more likely to fire; a negative bias makes it more conservative.

**Weighted Sum**  
The sum of all inputs multiplied by their corresponding weights: $\sum w_i x_i$.

**Activation ($z$)**  
The pre-activation value computed as the weighted sum plus the bias: $z = \sum w_i x_i + b$. Represents the total input signal before the activation function is applied.

**Activation Function**  
A function that transforms the activation into the final output. Our perceptron uses the **step function**: output 1 if $z > 0$, else 0.

**Output ($\hat{y}$)**  
The perceptron's final binary decision: 1 if the activation exceeds zero, 0 otherwise.

**Training Data**  
A set of labeled examples $(\mathbf{x}, y)$ used to train the perceptron. Each example consists of input features and the corresponding correct output.

**Prediction**  
The output the perceptron produces for a given input before seeing the correct answer.

**Error**  
The difference between the correct label and the prediction ($y - \hat{y}$). Tells us which direction to adjust weights: +1 if prediction too low, -1 if too high, 0 if correct.

**Learning Rate ($\eta$)**  
A small number (typically 0.1) that controls how much we adjust weights on each update. Too large and we overshoot; too small and learning takes forever.

**Epoch**  
One complete pass through all training examples. Multiple epochs are usually needed for the perceptron to converge.

**Update Rule**  
The formula for adjusting weights based on error: $w_i \leftarrow w_i + \eta \cdot (y - \hat{y}) \cdot x_i$. Moves weights in the direction that reduces future errors.

**Decision Boundary**  
The dividing line (or hyperplane) that separates the region where the perceptron outputs 1 from where it outputs 0. For a perceptron, this is always a straight cut through the input space.

**Linear Separability**  
A property of a classification problem where the two classes can be perfectly separated by a straight line (or hyperplane in higher dimensions). Perceptrons can only solve linearly separable problems.

**Hyperplane**  
The generalization of a decision boundary to any number of dimensions. In 1D it's a point, in 2D it's a line, in 3D it's a plane, and in higher dimensions it's a hyperplane.

**Perceptron Convergence Theorem**  
The mathematical guarantee that if a linearly separable solution exists, the perceptron learning algorithm will find it in a finite number of steps.

**Binary Classification**  
The task of classifying inputs into one of two categories. The perceptron outputs 1 for the positive class and 0 for the negative class.
