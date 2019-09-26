损失函数
========

L2Loss均方误差
>>>>>>>>>>>>>>

l2loss是真实值与预测值的差值的平方然后求和平均。通过平方的形式便于求导，所以常被用作线性回归的损失函数。

::

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = _reshape_like(F, label, pred)
        loss = F.square(label - pred)
        loss = _apply_weighting(F, loss, self._weight / 2, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)



