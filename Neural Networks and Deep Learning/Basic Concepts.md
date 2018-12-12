##Neural Network

x -> "neuron" -> y

ReLU: rectified linear unit 

###Supervised learning with Neural Network

####Types of NN

- standard NN
- image -> CNN
- sequence of data, e.g. audio, language translation -> (RNN recurrent)
- more advanced -> custome / hybrid

####Types of Data

- Structured
- Unstructured e.g. audio, text, image



---------------



## Binary Classification

e.g. (1) Cat or (0) Non Cat

image RGB 64x64 -> 

x = [……] , size n = 64x64x3

###Notations

$$ (x, y),  x \in R^{n_x}, y \in \lbrace 0,1\rbrace $$

m training examples: $$ \lbrace (x^{(1)}, y^{(1)}),(x^{(2)}, y^{(2)}), ... ,(x^{(m)}, y^{(m)}) \rbrace  $$

$$m = m_{train}$$ number of training examples

$$m_{test}$$ = number of test examples
$$
X =\begin{bmatrix}
x^{(1)} & x^{(2)} & ... & x^{(m)}
\end{bmatrix} = 
\begin{bmatrix}
x^{(1)T} \\ x^{(2)T} \\ ... \\ x^{(m)T}
\end{bmatrix}  (transpose, less used)
$$
width = m, height = $$n_x$$

$$X \in R^{n_x\times m}$$ 	X.shape = ($$n_x, m$$)
$$
Y =\begin{bmatrix}
y^{(1)} & y^{(2)} & ... & y^{(m)}
\end{bmatrix}
$$
$$Y \in R^{1\times m}$$	Y.shape = ($$1, m$$)



## Logistic Regression

Given x , want $$\hat{y} = P(y=1|x), x \in R^{n_x}, 0\leq\hat{y}\leq1$$ 

Parameters: $$w \in R^{n_x}, b \in R$$

Output: $$\hat{y} = \sigma(w^Tx+b)$$

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

If z large, $$\sigma(z) =1$$

If z large negative, $$\sigma(z) =0$$

Alternative notation (not used in the course):

$$x_0 = 1, x \in R^{n_x+1}$$
$$
\theta = 
\begin{bmatrix}
\theta_0 \\
\theta_1\\
\theta_2\\
...\\
\theta_{n_x}
\end{bmatrix}
$$

$$
b = \theta_0, w = \begin{bmatrix}
\theta_1\\
\theta_2\\
...\\
\theta_{n_x}
\end{bmatrix}
$$

###Logistic Regression Cost Function

 $$\hat{y} = \sigma(w^Tx+b)$$, where $$\sigma(z) = \frac{1}{1+e^{-z}}$$

Given $$ \lbrace (x^{(1)}, y^{(1)}),(x^{(2)}, y^{(2)}), ... ,(x^{(m)}, y^{(m)}) \rbrace  $$, want $${\hat{y}}^{(i)}\approx y^{(i)}$$, i is the ith trainning sample

Loss (error) function:

$$L(\hat{y}, y) = -(ylog\hat{y}+(1-y)log(1-\hat{y}))$$

If y = 1: $$L(\hat{y}, y) = -log\hat{y}$$  $$\leftarrow$$ want $$log\hat{y}$$ large, want $$\hat{y}$$ large

If y = 0: $$L(\hat{y}, y) = -log(1-\hat{y})$$ $$\leftarrow$$ want $$log(1-\hat{y})$$ large, want $$\hat{y}$$ small

Cost function:

$$J(w, b) = \frac{1}{m}\sum_{i=0}^mL({\hat{y}}^{(i)}, y^{(i)}) = -\frac{1}{m}\sum_{i=0}^m[y^{(i)}log{\hat{y}}^{(i)}+(1-y^{(i)})log(1-{\hat{y}}^{(i)})]$$



###Gradient Descent

Want to find w, b that minimize J(w,b)

Repeat{

$$w := w-\alpha\frac{dJ(w,b)}{dw}$$

$$ b := b-\alpha\frac{dJ(w,b)}{db}$$

alternatively —  $$w := w-\alpha dw, \alpha$$ is learning rate

}



### Computation Graph

$$J(a,b,c) = 3(a+bc)$$

$$u = bc$$

$$v = a+u$$

$$ J=3v$$





### Logistic Regression Gradient Descent

Logistic regression recap

$$z = w^Tx+b$$

$$\hat{y} = a = \sigma (z)$$

$$L(a, y) = -(ylog(a)+(1-y)log(1-a))$$

say we have $$x_1, w_1, x_2, w_2, b$$ 

then $$z = w_1x_1+w_2x_2+b, \hat{y} = a = \theta(z) \to L(a,y)$$

Steps:

1. $$da = \frac{dL(a, y)}{da} = -\frac{y}{a}+\frac{1-y}{1-a}$$
2. $$dz = \frac{dL}{dz} = \frac{dl}{da}\cdot\frac{da}{dz}= a-y$$
3. $$dw_1 = \frac{dL}{dw_1}=x_1\cdot dz,   dw_2 = \frac{dL}{dw_2}=x_2\cdot dz $$
4. 







