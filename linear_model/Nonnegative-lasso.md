# Nonnegative-lasso

Recalling that for fixed $\lambda$ nonnegative-lasso $\hat{\beta}$ is defined by:

**$\hat{\beta} \in argmin||Y-X\beta||_2^2+\lambda\sum_{i=1}^p\beta_i$, subject to $\beta \geq0$**

In our model , we can write that:

**$minimizeF(\beta) = \beta'(X'X)\beta+(\lambda1_n-2X'Y)'\beta$, subject to $\beta\geq0$**

Let $A=X'X, b=(\lambda1_n-2X'Y)'$

Then we set $$A_{ij}^+=\begin{cases} 
​		A_{ij}, & if A_{ij}>0\\ 
​		0, & else 
​	\end{cases}$$ and $$A_{ij}^-=\begin{cases} 
​		|A_{ij}|, & if A_{ij}<0\\ 
​		0, & else 
​	\end{cases}$$



And let $F_a(\beta)=\frac{1}{2}\beta'A^+\beta, F_b(\beta)=b'\beta, F_c(\beta)=\frac{1}{2}\beta'A^-\beta$, and $a_i=\frac{\partial F_a}{\partial \beta_i}=(A^+\beta)_i, c_i=\frac{\partial F_c}{\partial \beta_i}=(A^-\beta)_i$

The iterative steps are:

**$\beta_i^{(m+1)}:=[\frac{-b_i+\sqrt{b_i^2+4a_ic_i}}{2a_i}]\beta_i^{(m)}$**

