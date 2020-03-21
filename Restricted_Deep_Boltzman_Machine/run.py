from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''

    hidden_nodes_list = np.array([200, 300, 400, 500])
    recon_errors = np.zeros((len(hidden_nodes_list), 20))
    
    print ("\nStarting a Restricted Boltzmann Machine..")
    for idx, ndim in enumerate(hidden_nodes_list):
        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                         ndim_hidden=ndim,
                                         is_bottom=True,
                                         image_size=image_size,
                                         is_top=False,
                                         n_labels=10,
                                         batch_size=10
    )
        recon_errors[idx] = rbm.cd1(visible_trainset=train_imgs, n_iterations=20)

    x = np.arange(0, 20, 1)
    for i in range(len(hidden_nodes_list)):
        plt.plot(x, recon_errors[i], label='Hidden' + str(hidden_nodes_list[i]))
        plt.xlabel('Epochs')
        plt.ylabel('Reconstruction error')
        plt.title('Recon loss vs Hidden layers')
        plt.legend()
    plt.show()

    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")

    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
    )

    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=2000)

    dbn.recognize(train_imgs, train_lbls)

    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")

    ''' fine-tune wake-sleep training '''

    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=2000)

    dbn.recognize(train_imgs, train_lbls)

    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="dbn")
