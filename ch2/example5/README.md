# Adams-Bashforth、Adams-Moulton 和 Gear 方法数值分析

本文档展示了 Adams-Bashforth（显式）、Adams-Moulton（隐式）和 Gear 方法的数值分析结果，包括数值解比较、误差分析和收敛性分析。

## 数值解比较

<div style={{ display: 'flex', justifyContent: 'center', gap: '2%', marginTop: '10px' }}>
  <figure style={{ width: '49%', textAlign: 'center', margin: 0 }}>
    <img
      src="https://github.com/FEMATHS/Example/blob/main/ch2/example5/1.png?raw=true"
      alt="Adams-Bashforth方法数值解比较"
      style={{ width: '100%' }}
    />
    <figcaption style={{ fontSize: '90%', color: 'black', fontStyle: 'Times New Roman', marginTop: '4px' }}>
      图 1：Adams-Bashforth方法（1-5阶）与精确解的比较
    </figcaption>
  </figure>
  <figure style={{ width: '49%', textAlign: 'center', margin: 0 }}>
    <img
      src="https://github.com/FEMATHS/Example/blob/main/ch2/example5/4.png?raw=true"
      alt="Adams-Moulton方法数值解比较"
      style={{ width: '100%' }}
    />
    <figcaption style={{ fontSize: '90%', color: 'black', fontStyle: 'Times New Roman', marginTop: '4px' }}>
      图 2：Adams-Moulton方法（1-5阶）与精确解的比较
    </figcaption>
  </figure>
</div>

<div style={{ display: 'flex', justifyContent: 'center', gap: '2%', marginTop: '10px' }}>
  <figure style={{ width: '49%', textAlign: 'center', margin: 0 }}>
    <img
      src="https://github.com/FEMATHS/Example/blob/main/ch2/example5/6.png?raw=true"
      alt="Gear方法数值解比较"
      style={{ width: '100%' }}
    />
    <figcaption style={{ fontSize: '90%', color: 'black', fontStyle: 'Times New Roman', marginTop: '4px' }}>
      图 3：Gear方法（1-6阶）与精确解的比较
    </figcaption>
  </figure>
  <figure style={{ width: '49%', textAlign: 'center', margin: 0 }}>
    <img
      src="https://github.com/FEMATHS/Example/blob/main/ch2/example5/8.png?raw=true"
      alt="三种方法4阶比较"
      style={{ width: '100%' }}
    />
    <figcaption style={{ fontSize: '90%', color: 'black', fontStyle: 'Times New Roman', marginTop: '4px' }}>
      图 4：三种4阶方法的直接比较（AB4 vs AM4 vs Gear4）
    </figcaption>
  </figure>
</div>

## 误差分析

<div style={{ display: 'flex', justifyContent: 'center', gap: '2%', marginTop: '10px' }}>
  <figure style={{ width: '49%', textAlign: 'center', margin: 0 }}>
    <img
      src="https://github.com/FEMATHS/Example/blob/main/ch2/example5/2.png?raw=true"
      alt="Adams-Bashforth方法误差分析"
      style={{ width: '100%' }}
    />
    <figcaption style={{ fontSize: '90%', color: 'black', fontStyle: 'Times New Roman', marginTop: '4px' }}>
      图 5：Adams-Bashforth方法（1-5阶）的误差分析（对数坐标）
    </figcaption>
  </figure>
  <figure style={{ width: '49%', textAlign: 'center', margin: 0 }}>
    <img
      src="https://github.com/FEMATHS/Example/blob/main/ch2/example5/5.png?raw=true"
      alt="Adams-Moulton方法误差分析"
      style={{ width: '100%' }}
    />
    <figcaption style={{ fontSize: '90%', color: 'black', fontStyle: 'Times New Roman', marginTop: '4px' }}>
      图 6：Adams-Moulton方法（1-5阶）的误差分析（对数坐标）
    </figcaption>
  </figure>
</div>

<div style={{ display: 'flex', justifyContent: 'center', gap: '2%', marginTop: '10px' }}>
  <figure style={{ width: '49%', textAlign: 'center', margin: 0 }}>
    <img
      src="https://github.com/FEMATHS/Example/blob/main/ch2/example5/7.png?raw=true"
      alt="Gear方法误差分析"
      style={{ width: '100%' }}
    />
    <figcaption style={{ fontSize: '90%', color: 'black', fontStyle: 'Times New Roman', marginTop: '4px' }}>
      图 7：Gear方法（1-6阶）的误差分析（对数坐标）
    </figcaption>
  </figure>
  <figure style={{ width: '49%', textAlign: 'center', margin: 0 }}>
    <img
      src="https://github.com/FEMATHS/Example/blob/main/ch2/example5/9.png?raw=true"
      alt="三种方法4阶误差比较"
      style={{ width: '100%' }}
    />
    <figcaption style={{ fontSize: '90%', color: 'black', fontStyle: 'Times New Roman', marginTop: '4px' }}>
      图 8：三种4阶方法的误差比较（对数坐标）
    </figcaption>
  </figure>
</div>

## 收敛性分析

<div style={{ display: 'flex', justifyContent: 'center', gap: '2%', marginTop: '10px' }}>
  <figure style={{ width: '49%', textAlign: 'center', margin: 0 }}>
    <img
      src="https://github.com/FEMATHS/Example/blob/main/ch2/example5/3.png?raw=true"
      alt="Adams-Bashforth方法收敛性分析"
      style={{ width: '100%' }}
    />
    <figcaption style={{ fontSize: '90%', color: 'black', fontStyle: 'Times New Roman', marginTop: '4px' }}>
      图 9：Adams-Bashforth方法（1-5阶）的收敛性分析
    </figcaption>
  </figure>
  <figure style={{ width: '49%', textAlign: 'center', margin: 0 }}>
    <img
      src="https://github.com/FEMATHS/Example/blob/main/ch2/example5/10.png?raw=true"
      alt="Adams-Moulton方法收敛性分析"
      style={{ width: '100%' }}
    />
    <figcaption style={{ fontSize: '90%', color: 'black', fontStyle: 'Times New Roman', marginTop: '4px' }}>
      图 10：Adams-Moulton方法（1-5阶）的收敛性分析
    </figcaption>
  </figure>
</div>

<div style={{ display: 'flex', justifyContent: 'center', gap: '2%', marginTop: '10px' }}>
  <figure style={{ width: '49%', textAlign: 'center', margin: 0 }}>
    <img
      src="https://github.com/FEMATHS/Example/blob/main/ch2/example5/11.png?raw=true"
      alt="Gear方法收敛性分析"
      style={{ width: '100%' }}
    />
    <figcaption style={{ fontSize: '90%', color: 'black', fontStyle: 'Times New Roman', marginTop: '4px' }}>
      图 11：Gear方法（1-6阶）的收敛性分析
    </figcaption>
  </figure>
  <figure style={{ width: '49%', textAlign: 'center', margin: 0 }}>
    <img
      src="https://github.com/FEMATHS/Example/blob/main/ch2/example5/12.png?raw=true"
      alt="三种方法4阶收敛性比较"
      style={{ width: '100%' }}
    />
    <figcaption style={{ fontSize: '90%', color: 'black', fontStyle: 'Times New Roman', marginTop: '4px' }}>
      图 12：三种4阶方法的收敛性比较
    </figcaption>
  </figure>
</div>

## 方法说明

### Adams-Bashforth 方法（显式）

- 显式多步方法，使用前几个时间点的函数值
- 计算效率高，但稳定性相对较差
- 支持 1-5 阶精度

### Adams-Moulton 方法（隐式）

- 隐式多步方法，需要迭代求解
- 稳定性好，精度高，但计算成本较高
- 支持 1-5 阶精度

### Gear 方法

- 专门为刚性微分方程设计的隐式方法
- 具有很好的稳定性性质
- 支持 1-6 阶精度

## 测试问题

本分析使用的测试问题为：
$$\frac{du}{dt} = u - \frac{2t}{u}, \quad u(0) = 1$$

精确解为：$u(t) = \sqrt{1 + 2t}$

步长设置：$h = \frac{1}{2^4} = \frac{1}{16}$
