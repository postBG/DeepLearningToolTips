# Graphs and Sessions

Tensorflow = data flow graph를 그리고 그것을 실행하기 위한 session을 생성해 실행

## Dataflow graph

Dataflow graph의 장점

![dataflow graph](assets/tensors_flowing.gif)

*  **Parallelism.** 
  operation들간의 의존관계를 명시적으로 그래프로 나타냄으로써, 병렬로 실행가능한 부분이 어디인지 쉽게 판단할 수 있게 해준다.
*  **Distributed execution.**
  위와 같은 이유로 데이터의 흐름을 그래프로 표현하므로써, 여러 디바이스(CPU, GPU, and TPU)에 적절하게 분리시킬 수 있다.
*  **Compilation.**
  그래프를 분석하므로써 더 빠른 코드를 generate할 수 있다.
*  **Portability.**
  dataflow graph represenation은 언어독립적이므로 서로 다른 언어로 작성할 수도 있고, 저장과 복구도 용이해진다.

## What is a `tf.Graph`?

`tf.Graph`는 다음 두 가지의 정보를 가지고 있음

* **Graph Structure.**

  node와 edge가 어떻게 연결되어 있는지의 구조를 말함.

* **Graph collections.**

  Tensorflow는 `tf.Graph`의 메타데이터를 저장할 수 있는 메커니즘을 제공하고 있음. `tf.add_to_collection(key, value)`함수를 이용하면 이러한 메타데이터를 추가할 수 있고, `tf.get_collection`을 이용하면 메타데이터를 조회해볼 수 있음. 많은 tensorflow의 많은 부분들이 이미 이 기능을 사용하고 있다. 예를 들면, `tf.Variable`을 사용하면 자동으로 *global variables*와 *trainable variables*에 추가됨. 그리고 여기에 추가된 정보를 `tf.train.Saver`나 `tf.train.Optimizer`가 사용하게 됨.

## Building a `tf.Graph`

대부분의 Tensorflow 프로그램은 dataflow graph construction phase를 가진다.  이 phase에서는  Tensorflow의 API를 invoke함으로써 `tf.Operation`(node)과 `tf.Tensor`(edge) 객체를 만들고 그것을 `tf.Graph`에 추가하게 됨. 그리고 Tensorflow는 같은 context에서 모든 API에게 암묵적으로 **default graph**를 주게 됨.

예를 들면,

* `tf.constant(42.0)`을 호출하면, 

  * `42.0`을 produce하는 `tf.Operation`객체를 하나 생성하고,
  * 이를 **default graph**에 추가하고,
  * contant value(42.0)를 나타내는 `tf.Tensor`를 반환

* `tf.matmul(x, y)`를 호출하면.

  * `tf.Tensor`객체인 `x`, `y`의 value를 곱하는 `tf.Operation`객체를 하나 생성하고,
  * 이를 **default graph**에 추가하고,
  * multiplication의 result를 나타내는 `tf.Tensor`를 반환

* `v = tf.Variable(0)`을 실행하면,

  * `tf.Session.run`의 call들 사이(between `tf.Session.call` calls)에 유지되는(persist) writable tensor vaule를 지닌 `tf.Operation` 객체를 하나 생성하고,
  * 이를 **default graph**에 추가

* `tf.train.Optimizer.minimize`를 호출하면,

  * gradient를 계산하는 operation과 tensor를 **default graph**에 추가하고,
  * run될 때, 계산된 gradient를 variable들의 set에 적용하는 `tf.Operation`를 반환

  ​

대부분의 프로그램은 **default graph**만으로 충분하겠지만, 여러 가지 그래프를 다룰 필요성이 있을 경우 가능함.

## Naming operations

`tf.Graph`는 자기가 가지고 있는 `tf.Operions`에 **namespace**를 정의함. Tensorflow는 각 operation마다 unique한 이름을 붙이지만, descriptive한 이름을 붙이는 것이 나중에 이름을 이용한 기능을 사용하거나 디버깅할 때에 편하다. 

Tensorflow의 `tf.Operation`명명규칙:

* 모든 `tf.Operation`를 생성하거나 `tf.Tensor`를 반환하는 API는 `name`을 받을 수 있음.

  * `tf.constant(42.0, name="answer")`를 호출하면 `name`이 `"answer"` 인 `tf.Operator`를 생성하고, `name`이 `"answer:0"인 ` `tf.Tensor`를 반환

* 만약에 **default graph** 에 이미 존재하는 `name`으로 또 생성할 경우, 뒤에 자동으로 `"_1"`, `"_2"`를 붙인다.

* `tf.name_scope`함수를 이용하면, 특정 context에 속한 operation 이름에 name scope prefix를 붙일 수 있다.

  현재는 name scope가 `"/"`를 이용하여 붙으며, `name`과 마찬가지로 현재 context에 name_scope명이 중복된다면 자동으로 뒤에 `"_1"`, `"_2"`를 붙인다.

  ```python
  c_0 = tf.constant(0, name="c")  # => operation named "c"

  # Already-used names will be "uniquified".
  c_1 = tf.constant(2, name="c")  # => operation named "c_1"

  # Name scopes add a prefix to all operations created in the same context.
  with tf.name_scope("outer"):
    c_2 = tf.constant(2, name="c")  # => operation named "outer/c"

    # Name scopes nest like paths in a hierarchical file system.
    with tf.name_scope("inner"):
      c_3 = tf.constant(3, name="c")  # => operation named "outer/inner/c"

    # Exiting a name scope context will return to the previous prefix.
    c_4 = tf.constant(4, name="c")  # => operation named "outer/c_1"

    # Already-used name scopes will be "uniquified".
    with tf.name_scope("inner"):
      c_5 = tf.constant(5, name="c")  # => operation named "outer/inner_1/c"
  ```

graph visualizer는 visual complexity를 낮추기 위해 name scope를 이용하여 operation들을 group화 한다.



Tensorflow의 `tf.Tensor`의 명명규칙: `"<OP_NAME>:<i>"`

* `"<OP_NAME>"`: 이 tensor를 생성하는 `tf.Operation`의 이름
* `"<i>"`: operation이 생성한 output(=tensor)의 index



## Placing operations on different devices

skip! [link](https://www.tensorflow.org/programmers_guide/graphs#placing_operations_on_different_devices)



## Tensor-like Objects

API 사용성의 편의를 위해, tensorflow 함수들은 Tensor-like object들을 인자로 받아 암묵적으로 [`tf.convert_to_tensor`](https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor)를 이용해 `tf.Tensor`로 변환한다.



Tensor-like Objects의 종류들:

* `tf.Tensor`
* `tf.Variable`
* `numpy.ndarray`
* `list`(and lists of tensor-like objects)
* Scala Python types: `bool`, `flaot`, `int`, `str`



사용자가 [`tf.register_tensor_conversion_function`](https://www.tensorflow.org/api_docs/python/tf/register_tensor_conversion_function)를 이용하여 직접 정의할 수도 있음.

*만일 tensor-like object의 크기가 커서 계속 tf.Tensor로 변환하는 것이 부담된다면 직접 `tf.convert_to_tensor`를 호출하여 미리 `tf.Tensor`로 변환할 수 있음*



## Executing a graph in a `tf.Session`

