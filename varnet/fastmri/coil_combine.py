"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
'''
应该是一个与 MRI（磁共振成像）相关的自定义或外部库，从代码后续的使用来看，它可能提供了一些针对 MRI 数据处理的特定函数，
例如处理复数数据相关的操作（像 complex_abs_sq 函数，从名称推测是用于计算复数的模的平方等操作，后面会看到具体使用情况），
方便在 MRI 数据相关的计算任务中调用，以实现如计算 Root Sum of Squares（RSS）等针对 MRI 图像特征的计算功能

这两个函数都是围绕计算 Root Sum of Squares（RSS）值展开的，一个针对普通张量数据，一个针对复数张量数据，
在 MRI 数据处理等相关应用场景中，它们是进行数据特征提取和整合的重要计算工具，
为后续更复杂的分析和处理操作提供基础的数据表示形式。
'''
import fastmri

'''
这个函数用于计算输入张量 data 的 Root Sum of Squares（RSS，均方根）值。在 MRI 数据处理等场景中，
RSS 常用于对多线圈（coil）数据进行整合等操作，这里假定传入的维度参数 dim 对应的维度就是线圈维度，
按照这个维度来进行相关的计算，最终返回计算得到的 RSS 值，以体现数据在特定维度上的一种综合特征。

data：是一个 torch.Tensor 类型的输入参数，代表需要计算 RSS 值的张量数据，它可以是任意维度的张量，
具体维度结构取决于实际应用场景中的数据表示形式，比如在 MRI 数据里可能包含了批量大小、线圈数量、图像的高度、宽度等多个维度信息。
dim：是一个整数类型的参数，默认值为 0，用于指定在计算 RSS 时沿着哪个维度进行操作，根据函数注释中的说明，
这里默认假设这个维度就是线圈维度，意味着在这个维度上对数据进行相关的求和、平方等计算来得到 RSS 值，
调用者可以根据实际数据中线圈维度的真实位置传入相应的整数值来改变计算的维度。
'''
def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    '''
    函数体的核心计算逻辑是 torch.sqrt((data**2).sum(dim))，它按照以下步骤进行操作：
    data**2：首先对输入张量 data 的每个元素进行平方操作，这是计算 RSS 的第一步，即先求出每个元素的平方值，
    得到一个与 data 维度相同的新张量，新张量中的每个元素都是原张量对应元素的平方。
    (data**2).sum(dim)：接着对上一步得到的平方后的张量沿着指定的维度 dim 进行求和操作，
    这样就把指定维度上的所有元素的平方值累加起来，例如，如果 dim 为 0（假设这个维度是线圈维度，且数据维度结构为 
    [线圈数量, 图像高度, 图像宽度]），那么就会对每个线圈对应的图像位置上的元素平方和进行累加，
    得到一个维度少了 dim 对应的那一维的新张量（在这里就是变成了 [图像高度, 图像宽度] 的维度结构），
    这个求和后的张量体现了在指定维度上的平方和信息。
    torch.sqrt((data**2).sum(dim))：最后对求和后的结果再进行开方操作，使用 torch.sqrt 函数求出平方根，
    得到的最终结果就是在指定维度（假设为线圈维度）上的 Root Sum of Squares（RSS）值，这个值综合了指定维度上各元素的信息，
    在 MRI 数据处理中可以用于例如衡量多线圈数据在某个位置上的整体强度等情况。
    '''
    return torch.sqrt((data**2).sum(dim))

'''
此函数专门用于计算复数输入张量 data 的 Root Sum of Squares（RSS）值，同样假定传入的维度参数 dim 对应的维度是线圈维度，
和前面 rss 函数类似，也是为了在 MRI 数据处理等涉及复数表示的场景中（比如 k - 空间数据、复数形式的图像数据等）
对多线圈的复数数据进行整合，通过特定的计算得到体现整体特征的 RSS 值并返回。
data：是一个 torch.Tensor 类型的输入参数，不过这里它代表的是包含复数数据的张量，在 MRI 相关应用中，
复数数据常用来表示图像在频域（如 k - 空间）或者其他需要同时表示幅度和相位信息的情况，
其维度结构同样取决于具体的数据组织形式，可能包含线圈数量等相关维度信息。
dim：整数类型参数，默认值为 0，用于指定沿着哪个维度进行 RSS 计算，默认假设是线圈维度，
作用和 rss 函数中的 dim 参数一致，可根据实际数据中线圈维度的真实位置进行调整。
'''
def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    '''
    函数体核心计算逻辑是 torch.sqrt(fastmri.complex_abs_sq(data).sum(dim))，具体步骤如下：
    fastmri.complex_abs_sq(data)：首先调用 fastmri 库中的 complex_abs_sq 函数（
    从名称推测是用于计算复数的模的平方的函数，其具体实现取决于 fastmri 库内部代码，
    但功能上就是对输入的复数张量 data 计算每个复数元素对应的模的平方值），得到一个新的张量，
    这个新张量中每个元素是原复数张量对应元素的模的平方，其维度和 data 相同，
    这样就把复数数据转换为了实数值表示的每个元素的模的平方形式，方便后续计算。
    fastmri.complex_abs_sq(data).sum(dim)：接着对上一步得到的张量沿着指定的维度 dim 进行求和操作，
    原理和 rss 函数中类似，将指定维度（假设为线圈维度）上各元素的模的平方值累加起来，
    得到一个维度减少了 dim 对应的那一维的新张量，这个新张量体现了在指定维度上复数数据模的平方和信息。
    torch.sqrt(fastmri.complex_abs_sq(data).sum(dim))：最后对求和后的结果再进行开方操作，
    使用 torch.sqrt 函数求出平方根，得到的最终结果就是在指定维度（假设为线圈维度）上复数输入数据的
    Root Sum of Squares（RSS）值，通过这样的计算，对复数形式的多线圈数据进行了整合，
    提取出了在特定维度上的一种综合特征表示，常用于后续的 MRI 图像分析、重建等相关任务中的数据处理环节。
    '''
    return torch.sqrt(fastmri.complex_abs_sq(data).sum(dim))
