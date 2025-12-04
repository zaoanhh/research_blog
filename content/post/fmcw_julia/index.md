---
title: 基于 Julia 的 FMCW 雷达工作原理仿真
date: 2025-12-04T20:43:40+08:00
tags:
  - Julia
  - radar
image: 
categories:
  - code
---

本文介绍了fmcw测距的julia仿真。

## FFCW 测距

雷达通常也被叫做调频连续波（FMCW）雷达是一个使用频率调制来测量目标的距离的系统。在频率调制中，电磁波的频率随时间线性增加。或者说，发射频率会以恒定速率改变。这种频率随着时间线性增加的信号被称为 chirp。FMCW 系统测量发射信号和反射信号频率的瞬时差异δf，这直接和反射的 chirp 的时间差成比例。这个时间差能用来计算目标的距离。

![](4100650ab06cc0952fe0d20e5bfc07ca7d94e3e6cc3364bde2cedb456bbc6667.png)

上图（左）显示了一个 chirp 频率随时间变化的表现，右边显示频率随时间线性增加的 chirp 的幅度随时间变化图。

雷达前面的单目标产生的中频信号是一个固定频率音调，这个频率由下式给出：

$$
IF = \frac{2 \beta d}{c}
$$

这里 $d$ 是目标到雷达的距离，单位 m，$c$ 是光速，m/s，$\beta$ 是 chirp 斜率，由 chirp 持续时间内带宽的变化率得到。因此，我们可以对中频信号做 FFT 得到频率，再通过测量频率来计算距离。

## 测距仿真

1. 设置雷达参数

``` julia
# 导入包
using FFTW, SignalAnalysis
using CairoMakie, LinearAlgebra 
using DSP
CairoMakie.activate!(type="svg")
function myfig(;length_div_width=4/2,linewidth=28,fig_width_div_linewidth=0.5,scale=2.0)
    cm_to_pt = 28.3464566929134
    fig_length = floor(Int,linewidth * fig_width_div_linewidth * cm_to_pt * scale)
    fig_width = floor(Int,fig_length / length_div_width)
    return Figure(size =(fig_length,fig_width),pt_per_unit = 1)
end
```

``` julia
#| output: false
# 定义常量
maxR = 200.0      # 最大范围 (m)
rangeRes = 1.0    # 范围分辨率 (m)
maxV = 70.0       # 最大速度 (m/s)
# const fc = 77e9         # 载波频率 (Hz)
fc = 0.0         # 载波频率 (Hz)
c = 3e8           # 光速 (m/s)

# 目标参数
r0 = 100.0        # 目标距离 (m)
v0 = 70.0         # 目标速度 (m/s)

# 计算带宽和其他参数
B = c / (2 * rangeRes)  # 带宽 (Hz)
Tchirp = 2.5 * 2 * maxR / c  # Chirp时间 (s)
endle_time = 6.3e-6     # 结束时间 (s)
slope = B / Tchirp      # Chirp斜率 (Hz/s)
f_IFmax = (slope * 2 * maxR) / c  # 最大IF频率 (Hz)
f_IF = (slope * 2 * r0) / c  # 当前IF频率 (Hz)

# ADC采样点数和其他参数
Nd = 1          # Chirp数量
Nr = 1024         # ADC采样点数
vres = (c / fc) / (2 * Nd * (Tchirp + endle_time))  # 速度分辨率 (m/s)
Fs = Nr / Tchirp        # 采样率 (samples/s)
B
```

2. 发射信号, 假设发射的是 cos 信号，频率随时间线性变化，假设初始相位为0。发射信号为 $cos(2 \pi  \int_0^t f(t))$，其中 $f(t)$ 是随时间变化的频率, $f = f_c + \beta t$

``` julia
t = range(0, stop=Nd*Tchirp, length=Nr*Nd)  # 时间向量
dt = t[2] - t[1]  # 时间步长
freq = fc .+ mod.(slope .* t,B)  # 发射信号的频率
angle_freq = [sum(freq[1:i].*dt) for i in axes(t,1)]  # 发射信号的角度频率

# 生成发射波形
Tx = cos.(2 * pi * angle_freq)  # 发射波形

fig_tx = myfig(length_div_width=3,linewidth=28)
ax_tx = Axis(fig_tx[1, 1], xlabel="Time (s)", ylabel="Amplitude",title="发射信号")
lines!(ax_tx, t[1:Nr], Tx[1:Nr])

ax_freq = Axis(fig_tx[1, 2], xlabel="Time (s)", ylabel="Frequency (GHz)", title="频率随时间的变化")
lines!(ax_freq, t[1:Nr], freq[1:Nr]./1e9)
fig_tx
```

3.接收信号, 接收波形可以从发射波形和时延计算。对于目标距离 $r_0$，信号时延为 $t_d=2r_0/c$，接收信号为 $cos(2 \pi  \int_0^t f(t - t_d))$。

``` julia
td = 2 * r0 / c  # 目标延迟时间
freq_rx = [t_now > td ? fc .+ mod.(slope .* (t_now.-td),B) : fc for t_now in t]   # 接收信号的频率
angle_freq_rx = [sum(freq_rx[1:i].*dt) for i in axes(t,1)]  # 接收信号的角度频率
Rx = cos.(2 * pi .* angle_freq_rx)  # 接收信号
fig_rx = myfig(length_div_width=3,linewidth=28)
ax_rx = Axis(fig_rx[1, 1], xlabel="Time (s)", ylabel="Amplitude",title="接收信号")
lines!(ax_rx, t[1:Nr], Rx[1:Nr])

ax_freq = Axis(fig_rx[1, 2], xlabel="Time (s)", ylabel="Frequency (GHz)", title="频率随时间的变化")
lines!(ax_freq, t[1:Nr].+td, freq_rx[1:Nr]./1e9)
fig_rx
```

发射信号和接收信号的[时频图](https://arxiv.org/pdf/2101.06707)对比如下：

``` julia
# TX_stft = tfd(Rx[1:Nr], Spectrogram(nfft=64, window=hanning,noverlap=63),fs=Fs)
TX_stft = tfd(Tx[1:Nr], Wigner(nfft=256, smooth=10, method=:CM1980, window=hamming),fs=Fs)
RX_stft = tfd(Rx[1:Nr], Wigner(nfft=256, smooth=10, method=:CM1980, window=hamming),fs=Fs)
fig_stft = myfig(length_div_width=3,linewidth=28)
ax_tx_stft = Axis(fig_stft[1, 1], xlabel="Time (s)", ylabel="Frequency (GHz)",title="发射信号")
heatmap!(ax_tx_stft,TX_stft.time,TX_stft.freq ./1e9,TX_stft.power',colormap=:roma)

ax_rx_stft = Axis(fig_stft[1, 2], xlabel="Time (s)", ylabel="Frequency (GHz)",title="接收信号")
heatmap!(ax_rx_stft,RX_stft.time,RX_stft.freq ./ 1e9,RX_stft.power',colormap=:roma)
fig_stft
```

4. 计算中频信号。中频信号是接收信号与发射信号相乘，然后低通滤波的结果，为 $\text{IF} = cos(2\pi \int_0^t (f(t-t_d)-f(t)))$，注意到 **在实际的毫米波设备上，一个chirp信号发完之后会留出很长的空闲时间**，在这里我们只关注一个chirp信号的情况

``` julia
end_index = round(Int,(maximum(t)-td)/maximum(t)*Nr)
IF_angle_freq_all = [sum((freq[1:i].-freq_rx[1:i]).*dt) for i in axes(t,1)]
IF_angle_freq = IF_angle_freq_all[end_index+1:Nr]  # 中频信号的角度频率

IFx = cos.(2 * pi .* IF_angle_freq)  # 中频信号
fig_ifx = myfig(length_div_width=3,linewidth=28)
ax_ifx = Axis(fig_ifx[1, 1], xlabel="Time (s)", ylabel="Amplitude",title="接收信号")
lines!(ax_ifx, t[end_index+1:Nr], IFx)
IFX_fft = (abs.(fft([i>length(IFx) ? 0.0 : IFx[i] for i in 1:Nr])))
ax_ifx = Axis(fig_ifx[1, 2], xlabel="频率 (GHz)", ylabel="Amplitude", title="FFT ")
lines!(ax_ifx, IFX_fft[1:round(Int,Nr/2)]./Nr)
fig_ifx
```

``` julia
freq_res = findfirst(x->x==maximum(IFX_fft[1:round(Int,Nr/2)]),IFX_fft[1:round(Int,Nr/2)])
res = (freq_res / Nr ) * Tchirp *c
println("目标距离为: ",res," m")
```

## 多天线测角度

``` julia
xx = zeros(ComplexF64, 181)
θ = π /3
num = 30
xx[1:num] .= exp.((1:num) .* 1im .* sin(θ) .* π )
fig_test = myfig(length_div_width=3,linewidth=28)
ax_test = Axis(fig_test[1, 1], xlabel="Real", ylabel="Imaginary",title="复数")
lines!(ax_test, collect(-90:90),abs.(fftshift(fft(xx))))
fig_test
```

``` julia
res_aoa=asin((findfirst(abs.(fftshift(fft(xx))) .== maximum(abs.(fftshift(fft(xx)))))-90)/90) / π *180
println("目标角度为: ",res_aoa," °")
```

