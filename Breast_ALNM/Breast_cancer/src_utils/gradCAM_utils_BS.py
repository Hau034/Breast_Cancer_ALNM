import cv2
import numpy as np



class ActivationsAndGradients_BS:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        # 遍历我们需要输出的网络层结构
        for target_layer in target_layers:
            # 注册一个正向传播的钩子函数
            '''
            钩子函数：当网络进行正向传播的时候，当数据经过我们所指向的target层的时候，机会将数据创输给save_activation 函数
            
            '''
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                # 注册一个反向传播的钩子函数
                '''
                当梯度信息反传经过我们指定的target_layer层的时候，就会把数据传输给 save_gradient 函数
                '''
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:  # 旧版本使用（针对pytorch不同版本）
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        # 获取当前网络层结构的输出，复制给activation
        activation = output
        # 这个方法在 使用transformer架构的时候才会使用到
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        # 将activation 数据放到内存当中(detach切断梯度）
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]  # 第一个元素提取出来
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
            # 把获得的信息放在梯度的最前面
            '''
            正向传播从底层流向高层，反向传播从高层走向底层，如果把反向传播反过来放的话
            得到的梯度和和正向传播的顺序放置是一样的。
            '''
        self.gradients = [grad.cpu().detach()] + self.gradients

    # 正向传播过程
    def __call__(self, x1, x2):
        # 清空之前的信息
        self.gradients = []
        self.activations = []
        # 把 x 送进正向传播，出发我们注册的钩子函数，钩子函数自动把信息保存到
        # gradients 和 activations列表当中
        return self.model(x1, x2)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM_BS:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients_BS(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            # 如果只有一张图片，那么 loss即使等于最感兴趣类别的预测概率
            loss = loss + output[i, target_category[i]]  # target_category[i] 第i张图片所指定的类别（所感兴趣类别的预测值加到loss上去）

        return loss

    def get_cam_image(self, activations, grads):
        # 求梯度信息在 高度和宽度求均值，得到每一个通道的权重
        weights = self.get_cam_weights(grads)
        # 每一个通道的权重对激活层的参数进行加权
        weighted_activations = weights * activations
        # 求和
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor[0].size(-1), input_tensor[0].size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        # 正向传播收集的 激活值
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        # 反向传播手机的 梯度值
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        # 得到输入图片的高度与宽度
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        '''
        同时遍历激活层的输出和梯度(现在顺序是从前到后是一致的）
        '''
        for layer_activations, layer_grads in zip(activations_list, grads_list):
            # 对每一个需要计算的层在反向传播之后进行加权求和，
            cam = self.get_cam_image(layer_activations, layer_grads)
            # 将小于0的元素全部置为0
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            # 将cam和原图的大小输入到 scale_cam_image 方法中，对cam进行调整成 和原图一样的大小，以便后面叠加在原图上面
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    # 对所有特征层的信息进行融合
    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        '''
        正向传播得到输出的同时会得到 target_layers 的梯度和激活值
        '''
        bmode_tensor, swe_tensor = input_tensor
        # output = self.activations_and_grads(input_tensor)
        # 两个输入
        output = self.activations_and_grads(bmode_tensor, swe_tensor)

        i = 0

        if isinstance(target_category, int):  # 判断  target_category 传入的是否是int类型
            # 根据传入的 batch数量重新生成 target_category ，目的是一次性可以求得多个图像相对于一个类别的Grad-CAM
            target_category = [target_category] * input_tensor[i].size(0)

        if target_category is None:  # 如果不指定target_category，那么就会预测网络输出分数最大的那个索引
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor[i].size(0))
        # 清空历史的梯度信息
        self.model.zero_grad()
        # 得到还没有经过激活层，预测类别输出概率
        loss = self.get_loss(output, target_category)
        # 对感兴趣的类别输出预测的概率进行反向传播，会触发我们反向传播的钩子函数
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        # 返回和原图一样尺寸的 cam ，其中包含指定的每一个层（返回我们指定的每一个层结构的 cam数据）
        cam_per_layer = self.compute_cam_per_layer(input_tensor[i])
        # 将我们指定的所有网络层的cam进行融合，如果只有一张就不用了。
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


class GradCAM_BmodeSwe:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients_BS(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            # 如果只有一张图片，那么 loss即使等于最感兴趣类别的预测概率
            loss = loss + output[i, target_category[i]]  # target_category[i] 第i张图片所指定的类别（所感兴趣类别的预测值加到loss上去）

        return loss

    def get_cam_image(self, activations, grads):
        # 求梯度信息在 高度和宽度求均值，得到每一个通道的权重
        weights = self.get_cam_weights(grads)
        # 每一个通道的权重对激活层的参数进行加权
        weighted_activations = weights * activations
        # 求和
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        # 正向传播收集的 激活值
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        # 反向传播手机的 梯度值
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        # 得到输入图片的高度与宽度
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        '''
        同时遍历激活层的输出和梯度(现在顺序是从前到后是一致的）
        '''
        for layer_activations, layer_grads in zip(activations_list, grads_list):
            # 对每一个需要计算的层在反向传播之后进行加权求和，
            cam = self.get_cam_image(layer_activations, layer_grads)
            # 将小于0的元素全部置为0
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            # 将cam和原图的大小输入到 scale_cam_image 方法中，对cam进行调整成 和原图一样的大小，以便后面叠加在原图上面
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    # 对所有特征层的信息进行融合
    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor_b,input_tensor_s ,target_category=None):

        #(input_tensor_bmode, input_tensor_swe) = input_tensor
        if self.cuda:
            input_tensor_bmode = input_tensor_b.cuda()
            input_tensor_swe = input_tensor_s.cuda()
        else:
            input_tensor_bmode = input_tensor_b
            input_tensor_swe = input_tensor_s
        # 正向传播得到网络输出logits(未经过softmax)
        '''
        正向传播得到输出的同时会得到 target_layers 的梯度和激活值
        '''
        output = self.activations_and_grads(input_tensor_bmode, input_tensor_swe)
        if isinstance(target_category, int):  # 判断  target_category 传入的是否是int类型
            # 根据传入的 batch数量重新生成 target_category ，目的是一次性可以求得多个图像相对于一个类别的Grad-CAM
            target_category = [target_category] * input_tensor_bmode.size(0)

        if target_category is None:  # 如果不指定target_category，那么就会预测网络输出分数最大的那个索引
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor_bmode.size(0))
        # 清空历史的梯度信息
        self.model.zero_grad()
        # 得到还没有经过激活层，预测类别输出概率
        loss = self.get_loss(output, target_category)
        # 对感兴趣的类别输出预测的概率进行反向传播，会触发我们反向传播的钩子函数
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        # 返回和原图一样尺寸的 cam ，其中包含指定的每一个层（返回我们指定的每一个层结构的 cam数据）
        cam_per_layer = self.compute_cam_per_layer(input_tensor_bmode)
        # 将我们指定的所有网络层的cam进行融合，如果只有一张就不用了。
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    # 注意opencv使用的是BGR格式
    if use_rgb:
        # 转换成RGB的歌手
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # 把图像转换成渐变色之后，再对图像像素进行缩放到 0 - 1 之间
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    # 两种图片按照 1:1 的比例进行叠加
    cam = heatmap + img
    cam = cam / np.max(cam)  # 为了让数据不大于1，再次进行缩放
    return np.uint8(255 * cam)  # 再次乘上255，像素从0-255，返回


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h + size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w + size]

    return img
