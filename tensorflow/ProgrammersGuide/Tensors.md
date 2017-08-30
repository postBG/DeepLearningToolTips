# Tensors

### Tensor의 shape

* Graph를 만드는 중에 shape를 알기 위해서는 *tf.Tensor*의 shape 속성을 읽으면 된다.

* 만일 runtime 중 (따라서 Tensor의 형태가 fully-defined되어 있는 경우)에 shape를 얻는데에는 *tf.shape*를 이용하여 읽으면 된다.

  ```py
  # 이 코드는 my_matrix의 column과 같은 크기의 zero 텐서를 만드는 예제
  zeros = tf.zeros(tf.shape(my_matrix)[1])
  ```



### Tensor의 shape 변경하기

```python
rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])  # Reshape existing content into
                                                 # a 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])  #  Reshape existing content into a 3x20
                                       # matrix. -1 tells reshape to calculate
                                       # the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a
                                             #4x3x5 tensor

# Note that the number of elements of the reshaped Tensors has to match the
# original number of elements. Therefore, the following example generates an
# error because no possible value for the last dimension will match the number
# of elements.
yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # ERROR!
```



### Evaluating Tensors

* ```python
  constant = tf.constant([1, 2, 3])
  tensor = constant * constant
  print tensor.eval() # 반환값은 numpy
  ```

  위의 코드 블록과 같이 *Tensor.eval* 메소드를 사용하면 Tensor의 값을 계산할 수 있다.

  단,  *Tensor.eval* 메소드는 default tf.Session 이 활성화된 경우에만 작동한다.

* 위의 예시와는 다르게 context없이 계산 불가능할 수도 있는데, Tensor가 dynamic한 정보에 의존하는 경우가 그렇다. 예를 들면, *Placeholder* 를 사용하는 경우는 아래와 같이 계산해야한다.

  ```python
  p = tf.placeholder(tf.float32)
  t = p + 1.0
  t.eval()  # This will fail, since the placeholder did not get a value.
  t.eval(feed_dict={p:2.0})  # This will succeed because we're feeding a value
                             # to the placeholder.
  ```

  ​