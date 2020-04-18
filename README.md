# iLQR

Implementation of iLQR (Iterative Linear Quadratic Regulator), with example
application of swinging up a cart pole from an arbitrary initial state.

See [my blog post on
iLQR](https://harwiltz.github.io/control/2020/02/19/ilqr-without-obfuscation.html)
for more details.

## Usage

To reproduce the results shown in the blog post, execute the following command
from the project root:

```bash
python ilqr_cartpole_learn.py --horizon 12000 \
                              --iterations 50 \
                              --thresh 100 \
                              --grad_clip 30
```
