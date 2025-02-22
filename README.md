# deep-learning-ramp-up

Some deep learning projects to ramp up on the space.
Provided by Jacob Buckman.

Good reference materials

- https://d2l.ai/
- https://fleuret.org/public/lbdl.pdf
- https://karpathy.github.io/2015/05/21/rnn-effectiveness/
- https://www.youtube.com/watch?v=zjkBMFhNj_g
- https://www.youtube.com/@statquest

Suggested projects

Each of these should take between 1-5 days. A lot of the necessary info is available in the materials I linked above, but if you are lost on what one of these means or how to accomplish it, just reach out. Send me the code after each step is complete so I can check it & give feedback (I suggest just making a github repo and committing each step as a separate folder).

For these first 5, use synthetic data, e.g. choose a random input x & then apply some function f(x) to get the target output. Make a big dataset of examples like this for your training data. Don't worry about test sets.
1. Linear regression where input & output are each a single scalar, with stochastic gradient descent (minibatches), using numpy. (The trickiest part of this one is implementing backprop by hand.)
2. Same but using pytorch (torch.backward). Also, auto-render a plot of the results (steps of GD on x-axis, loss on y-axis).
3. Linear regression where input is a vector.
3. Classification where input is a vector, output is categorical (softmax & negative log likelihood loss).
4. Now use a feedforward layer instead of pure linear regression.
5. Now use N feedforward layers (deep learning!)
Now move to the MNIST dataset. Also, start to measure & plot the test-set loss, in addition to the train set.
6. Still just classification on vector inputs, but now on MNIST. (Flatten each input into a vector.)
7. Use the optax library & the Adam optimizer.
8. Change architecture to be a convnet.
9. Change architecture to be a resnet.
Now move to the shakespeare dataset from Karpathy.
10. Go back to a feedforward architecture. Each input should be a sequence of N words, output should be the (N+1)th word. (Classification, the classes are all the possible words in the vocabulary, negative log likelihood loss.)
11. Add a new evaluation mode: autoregressive sampling. Do this every time you evaluate the test set.
12. Change architecture to be a causal transformer. Input data is still N words, but output data is now N next-words (ie the same sequence but offset by 1).
Now, you will be going onto GPU.
13. Go back to your resnet classification-on-MNIST example, and make it run fast on a GPU. (This will mostly mean doing as much as possible as conv/matmuls, and making sure data loading is not a bottleneck.)
14. Switch from MNIST to ImageNet. Adjust your resnet setup by copying the literature. Do a multi-day run on ImageNet and see how you score, and compare to what other people have reported.
15. Go back to your transformer. Make it work on GPU, and try to match Karpathy's implementation and performance from the nanoGPT repo (on the shakespeare dataset).
16. Upgrade your code to work on multiple GPUs, get your hands on some, switch your dataset to openwebtext, and try to match Karpathy's implementation and performance (at GPT2 scale).
