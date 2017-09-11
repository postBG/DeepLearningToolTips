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

`tf.Session`은 client code(예를 들면 python code)와 tensorflow graph를 연결하는 역할을 한다.

`tf.Session`은 device에 접근하거나 `tf.Graph`를 cache하여 같은 연산을 효율적으로 반복할 수 있게 해준다.



### Creating a `tf.Session`

```python
# Create a default in-process session.
with tf.Session() as sess:
  # ...

# Create a remote session.
with tf.Session("grpc://example.org:2222"):
  # ...
```

`tf.Session`은 physical resource(네트워크 커넥션이나 GPU, CPU 등)들을 직접 소유하고 있기 때문에, 블록을 빠져나가면 알아서 종료되는 context manager(`with`문)과 함께 사용된다. 만일 context manager를 사용하고 싶지 않으면 명시적으로 `tf.Session.close`를 호출해줘야 함.

*__**Note**__ 더 High-level API들(예를 들면,  [`tf.train.MonitoredTrainingSession`](https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession) or [`tf.estimator.Estimator`](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator))들은 스스로 `tf.Session`을 생성하고 관리한다. 이러한 API는 __**target**__과 __**config**__를 입력받아 작동하는데 직접 인자로 넘길 수도 있고, [`tf.estimator.RunConfig`](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig)를 통해서 넘길 수도 있다.*

`tf.Session.__init__`는 3가지의 선택적인 인자를 받음:

* **target**
  * *Default:* 이 인자가 비어있으면(=default) local machine에 있는 device만 사용.
  * 하지만 `grpc://`와 같은 형식으로 device를 제공하는 Tensorflow Server를 지정할 수도 있다.
* **graph**
  * *Default:* 새 Session은 오직 현재의 default graph에 bound된다.
  * 만일 여러 개의 Graph들을 다루는 경우([Programming with multiple graphs](https://www.tensorflow.org/programmers_guide/programming-with-multiple-graphs) 참고)에는 session을 생성할 때 지정할 수 있다.
* **config** 
  * [`tf.ConfigProto`](https://www.tensorflow.org/api_docs/python/tf/ConfigProto)을 지정할 수 있게 해주는 옵션



### Using [`tf.Session.run`](https://www.tensorflow.org/api_docs/python/tf/Session#run) to execute operations

`tf.Session.run`은 `tf.Operation`과 `tf.Tensor`를 계산하는 주된 매커니즘이다. `tf.Session.run`에 하나 이상의 `tf.Operation`과 `tf.Tensor`를 넘기면 Tensorflow는 결과를 계산하는데 필요한 operation들을 실행한다.



`tf.Session.run`은 **fetch**들의 list를 요구하는데,  이 **fetch**는 `tf.Operation`, `tf.Tensor`, 혹은 위에서 설명한 tensor-like type이 될 수 있음.

이 **fetch**들은 먼저 return value를 결정하게 되며, 이 return value(=result)를 계산하기 위해 반드시 실행되어야할  **subgraph**를 전체 `tf.Graph` 추려내고 실행되도록 한다.

```python
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
init_op = w.initializer

with tf.Session() as sess:
  # Run the initializer on `w`.
  sess.run(init_op)

  # Evaluate `output`. `sess.run(output)` will return a NumPy array containing
  # the result of the computation.
  print(sess.run(output))

  # Evaluate `y` and `output`. Note that `y` will only be computed once, and its
  # result used both to return `y_val` and as an input to the `tf.nn.softmax()`
  # op. Both `y_val` and `output_val` will be NumPy arrays.
  y_val, output_val = sess.run([y, output])
```



`tf.Session.run`은 선택적으로 dictionary of **feeds**를 입력으로 받을 수 있다. 이 **feed**들은 그래프가 실행되는 시점에  `tf.Tensor`객체(주로 `tf.placeholder`)와 값을 연결하는 역할을 한다. 즉, `tf.Tensor`는 **feed**로 받은 값으로 대체되어 실행된다.

```python
# Define a placeholder that expects a vector of three floating-point values,
# and a computation that depends on it.
x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

with tf.Session() as sess:
  # Feeding a value changes the result that is returned when you evaluate `y`.
  print(sess.run(y, {x: [1.0, 2.0, 3.0]})  # => "[1.0, 4.0, 9.0]"
  print(sess.run(y, {x: [0.0, 0.0, 5.0]})  # => "[0.0, 0.0, 25.0]"

  # Raises `tf.errors.InvalidArgumentError`, because you must feed a value for
  # a `tf.placeholder()` when evaluating a tensor that depends on it.
  sess.run(y)

  # Raises `ValueError`, because the shape of `37.0` does not match the shape
  # of placeholder `x`.
  sess.run(y, {x: 37.0})
```



`tf.Session.run`은 호출에 대한 option을 줄 수 있는 `options`와 실행에 대한 meta정보를 얻을 수 있게 해주는  `run_metadata`라는 인자도 선택적으로 받을 수 있다. 예를 들어, 이 두가지 인자를 조합하여 'collect tracing information about the execution'를 할 수 있다.

```python
y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
  # Define options for the `sess.run()` call.
  options = tf.RunOptions()
  options.output_partition_graphs = True
  options.trace_level = tf.RunOptions.FULL_TRACE

  # Define a container for the returned metadata.
  metadata = tf.RunMetadata()

  sess.run(y, options=options, run_metadata=metadata)

  # Print the subgraphs that executed on each device.
  print(metadata.partition_graphs)

  # Print the timings of each operation that executed.
  print(metadata.step_stats)
```

