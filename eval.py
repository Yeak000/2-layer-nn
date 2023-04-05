import pickle as pkl
from train import load_mnist_data, mnist_dir, one_hot, forward_propagation

with open('ckpts/lr:0.7  h_s:200  l2_penalty:1.pkl', 'rb') as f:
    model = pkl.load(f)
data = load_mnist_data(mnist_dir)
test_images, test_labels = data[2], data[3]
y = one_hot(test_labels, 10)
pred_y, _ = forward_propagation(test_images, model)

