# Variables

A TensorFlow **variable** is the best way to represent shared, persistent state manipulated by your program.



`tf.Variable`은 op를 적용하므로써 값이 변화할 수 있는 tensor를 나타낸다. 또 `tf.Tensor`와는 다르게 `tf.Variable`은 특정한 `session.run`의 context에 속해있지 않다.



내부적으로 `tf.Variable`은 persistent tensor를 가지고 있다. 특정 op는 이 tensor의 값을 읽거나 고칠 수 있고, 이렇게 변화된 사항은 여러 `tf.Session`이 볼 수 있다.



## Creating a Variable

변수를 생성하는 가장 좋은 방법은 `tf.get_variable`함수를 사용하는 것이다. 이 함수는 name을 입력받는데, 이 이름은 여러 replica들이 이름을 통해 이 값에 접근하고, exporting과 checkpointing에 사용할 수 있도록 해준다. 

또 `tf.get_variable`을 이용하면 이전에 만들어진 이름이 같은 변수를 재사용해서 layer를 재사용하는 모델을 쉽게 정의할 수도 있다.



가장 단순하게 변수를 정의하기 위해서는 name과 shape를 넘겨주면 된다.

```python
my_variable = tf.get_variable("my_variable", [1, 2, 3])
```

이렇게 만들어진 변수는 default로 `tf.glorot_uniform_initializer`를 이용해 초기화된 tensor가 된다.



`dtype`을 선택적으로 명시할 수도 있다.

```python
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32, 
  initializer=tf.zeros_initializer)
```



또한 특정값으로 tensor를 초기화할 수도 있다. 주의할 점은 이때는 tensor가 직접 initializer에 입력받은 tensor의 shape를 이용하기 때문에 shape를 입력해서는 안된다.

```python
other_variable = tf.get_variable("other_variable", dtype=tf.int32, 
  initializer=tf.constant([23, 42]))
```

