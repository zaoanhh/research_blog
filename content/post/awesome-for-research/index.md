---
title: Awesome for research
# description: Welcome to Hugo Theme Stack
slug: hello-world
date: 2025-11-23 00:00:00+0000
image: cover.jpg
author: Fei Shang
categories:
    - Awesome
tags:
    - awesome
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---


## 绘图

- [python matplotlib style](https://github.com/garrettj403/SciencePlots/tree/master)
- Julia绘图，[Makie.jl](https://docs.makie.org/stable/)，[makie的示例](https://beautiful.makie.org/dev/)
- [matplotlib to tikz](https://github.com/nschloe/tikzplotlib)
- [julia的tikz绘图](https://github.com/JuliaTeX/PGFPlots.jl?tab=readme-ov-file)
- 图标、插画网站：[阿里巴巴](https://www.iconfont.cn/search/index?searchType=icon&q=%E9%A5%AE%E6%96%99&page=1&fromCollection=1&fills=&tag=)、[花瓣](https://huaban.com/)
- [控制绘图字号与正文一致的办法](https://github.com/zaoanhh/makie_theme_for_paper)
- [DeTikZify](https://github.com/potamides/DeTikZify)：手绘图转tikz


## 实验管理

- [MLflow](https://mlflow.org/docs/latest/)：MLflow 是一个开源平台，专为帮助机器学习从业者和团队应对机器学习流程的复杂性而构建。MLflow 专注于机器学习项目的完整生命周期，确保每个阶段都可管理、可追溯且可复现。
- [swanlab](https://github.com/SwanHubX/SwanLab)：一个开源、现代化设计的深度学习训练跟踪与可视化工具，同时支持云端/离线使用，适配30+主流框架，与实验代码轻松集成
- [DearDiary.jl](https://github.com/JuliaAI/DearDiary.jl): Julia 的轻量级但功能强大的机器学习实验跟踪工具。
- [pytorch lighting + hydra](https://zaoanhh.github.io/research_blog/p/pytorch-lightning--hydra-工程化实践)


## 文档

- [Sphinx](https://github.com/sphinx-doc/sphinx): 可以从python函数注释中生成文档
- [**Documenter.jl](https://documenter.juliadocs.org/stable/#Documenter.jl)：**可以从julia函数注释中生成文档
- [buzz](https://github.com/chidiwilliams/buzz/tree/main): 离线语音转写

## 写作 & slide

- [Quarto](https://quarto.org/): 一个markdown语言的扩展版，可以嵌入python、julia、r语言的执行，并且可以方便的把文件输出为pdf、latex、html、website等
- [Pluto.jl](https://plutojl.org/): 一个julia的notebook，相较于jupyter最大的区别是它会保持变量的全局一致性。在jupyter的不同cell中我们可以对一个变量赋予不同的值，甚至出现cel的交叉执行，这使得执行结果依赖cel的执行顺序。pluto避免了这一点，笔记本的输出是一致的。
- [tex-diary](https://github.com/lqiang67/tex-diary)：这是一个基于 LaTeX 的研究日志系统，用于创建、整理和编译每日 LaTeX 笔记，并可将其生成 PDF 文档和 HTML 博客。该系统采用极简的标签系统，方便用户进行内容管理。
- [Overleaf-Workshop](https://github.com/overleaf-workshop/Overleaf-Workshop)：在 VSCode 中打开 Overleaf (ShareLatex) 项目，并支持完全协作。
- [pdfpc](https://github.com/pdfpc/pdfpc)：pdfpc 是一款基于 GTK 的演示应用程序，它使用类似 Keynote 的多显示器输出功能，在演示过程中为演讲者提供元信息。它可以在一个屏幕上显示普通的演示窗口，同时在另一个屏幕上显示更全面的概览信息，例如下一张幻灯片的图片、演示剩余时间等等。pdfpc 处理的输入文件是 PDF 文档，而大多数现代演示软件都可以创建 PDF 文档。
- [pympress](https://github.com/Cimbali/pympress)：与pdfpc功能差不多
- [BeamerQt](https://github.com/acroper/BeamerQt)：BeamerQT 是一款用户友好的图形界面，旨在简化 Beamer 演示文稿的创建过程，无需手动编辑与幻灯片关联的 LaTeX 代码。它提供了一套全面的功能，允许用户定义布局、插入内容（包括文本、代码块和图像）以及配置主题的一些高级设置。BeamerQT 使 LaTeX 初学者和高级用户都能轻松创建精彩的演示文稿，并将注意力集中在内容创作而非代码编写上。
- [Fei’s beamer theme](https://github.com/zaoanhh/phd-thesis-beamer--template)

## 计算库

- [**sionna](https://github.com/NVlabs/sionna)：**Sionna™ 是一个基于 Python 的开源库，用于通信系统研究。
    
    它由以下软件包组成：
    
    - [Sionna RT——](https://nvlabs.github.io/sionna/rt/index.html)一款速度极快的独立式射线追踪器，用于无线电传播建模。
    - [Sionna PHY——](https://nvlabs.github.io/sionna/phy/index.html)无线和光通信系统的链路级仿真器
    - [Sionna SYS——](https://nvlabs.github.io/sionna/sys/index.html)基于物理层抽象的系统级模拟器
- [Turing.jl](https://github.com/TuringLang/Turing.jl)：Julia 中的概率编程和贝叶斯推理
- [MLJ.jl](https://github.com/JuliaAI/MLJ.jl)：MLJ（Julia 中的机器学习）是一个用 Julia 编写的工具箱，它提供了一个通用的接口和元算法，用于选择、调整、评估、组合和比较[200 多个用 Julia 和其他语言编写的机器学习模型](https://juliaai.github.io/MLJ.jl/stable/model_browser/#Model-Browser)。
- [Lux.jl](https://github.com/LuxDL/Lux.jl)：JuliaLang 中的优雅且高性能的深度学习，兼具 Julia 的优雅外形和 XLA 的性能
- [Associations.jl](https://github.com/JuliaDynamics/Associations.jl)：用于量化关联、独立性测试和数据因果推断的算法。
- [JuMP.jl](https://github.com/jump-dev/JuMP.jl)：JuMP 是一种嵌入在[Julia](https://julialang.org/)中的领域特定[数学优化建模语言。您可以访问](https://en.wikipedia.org/wiki/Mathematical_optimization)[jump.dev](https://jump.dev/)了解更多信息。
- [sciml](https://sciml.ai/): 可组合的开源软件，用于科学机器学习，支持微分编程。运用最新技术构建基于物理学的AI，使用便捷.

## 公开数据集
- [Deepsense](https://www.deepsense6g.net/): DeepSense 6G is a real-world multi-modal dataset that comprises coexisting multi-modal sensing and communication data, such as mmWave wireless communication, Camera, GPS data, LiDAR, and Radar, collected in realistic wireless environments
- [Capture24](https://github.com/OxWearables/capture24) The purpose of the CAPTURE-24 dataset is to serve as a training dataset for developing Human Activity Recognition (HAR) classifiers.
- [physionet](https://physionet.org/) 复杂生理信号数据集


## 科研观点

- 杨铮老师：[想法是怎样炼成的](http://tns.thss.tsinghua.edu.cn/~yangzheng/files/%E6%83%B3%E6%B3%95%E6%98%AF%E6%80%8E%E6%A0%B7%E7%82%BC%E6%88%90%E7%9A%84.pdf)