import os
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import scipy

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir

        self.losses = []
        self.dice_losses_seg_T1= []
        self.dice_losses_seg_T2 = []
        self.ce_losses_classi= []
        self.val_losses= []
        self.val_dice_losses_seg_T1= []
        self.val_dice_losses_seg_T2 = []
        self.val_ce_losses_classi= []

        self.train_dices_T1=[]
        self.train_dices1_T1 = []
        self.train_dices2_T1 = []
        self.train_dices3_T1 = []
        self.train_dices4_T1 = []
        self.train_dices5_T1 = []
        self.train_dices6_T1 = []

        self.val_dices_T1=[]
        self.val_dices1_T1 = []
        self.val_dices2_T1 = []
        self.val_dices3_T1 = []
        self.val_dices4_T1 = []
        self.val_dices5_T1 = []
        self.val_dices6_T1 = []

        self.train_dices_T2=[]
        self.train_dices1_T2 = []
        self.train_dices2_T2 = []

        self.val_dices_T2=[]
        self.val_dices1_T2 = []
        self.val_dices2_T2 = []

        self.train_accs= []
        self.val_accs = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, dice_loss_seg_T1,dice_loss_seg_T2,ce_loss_classi,val_loss,
                    val_dice_loss_seg_T1,val_dice_loss_seg_T2,val_ce_loss_classi):
    # def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.dice_losses_seg_T1.append(dice_loss_seg_T1)
        self.dice_losses_seg_T2.append(dice_loss_seg_T2)
        self.ce_losses_classi.append(ce_loss_classi)
        self.val_losses.append(val_loss)
        self.val_dice_losses_seg_T1.append(val_dice_loss_seg_T1)
        self.val_dice_losses_seg_T2.append(val_dice_loss_seg_T2)
        self.val_ce_losses_classi.append(val_ce_loss_classi)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "dice_loss_seg_T1.txt"), 'a') as f:
            f.write(str(dice_loss_seg_T1))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_ce_loss_classi.txt"), 'a') as f:
            f.write(str(ce_loss_classi))
            f.write("\n")
        with open(os.path.join(self.log_dir, "dice_loss_seg_T2.txt"), 'a') as f:
            f.write(str(dice_loss_seg_T2))
            f.write("\n")
        with open(os.path.join(self.log_dir, "val_ce_loss_classi.txt"), 'a') as f:
            f.write(str(val_ce_loss_classi))
            f.write("\n")
        with open(os.path.join(self.log_dir, "val_dice_loss_seg_T1.txt"), 'a') as f:
            f.write(str(val_dice_loss_seg_T1))
            f.write("\n")
        with open(os.path.join(self.log_dir, "val_dice_loss_seg_T2.txt"), 'a') as f:
            f.write(str(val_dice_loss_seg_T2))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.writer.add_scalar('dice_loss_seg_T1', dice_loss_seg_T1, epoch)
        self.writer.add_scalar('dice_loss_seg_T2', dice_loss_seg_T2, epoch)
        self.writer.add_scalar('ce_loss_classi', ce_loss_classi, epoch)
        self.writer.add_scalar('val_dice_loss_seg_T1', val_dice_loss_seg_T1, epoch)
        self.writer.add_scalar('val_dice_loss_seg_T2', val_dice_loss_seg_T2, epoch)
        self.writer.add_scalar('val_ce_loss_classi', val_ce_loss_classi, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_losses, 'coral', linewidth = 2, label='val loss')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.dice_losses_seg_T1, 'red', linewidth=2, label='T1 train seg dice loss')
        plt.plot(iters, self.val_dice_losses_seg_T1, 'coral', linewidth=2, label='T1 val seg dice loss')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Seg T1 Dice Loss')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "epoch_T1_seg_dice_loss.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.dice_losses_seg_T2, 'red', linewidth=2, label='T2 train seg dice loss')
        plt.plot(iters, self.val_dice_losses_seg_T2, 'coral', linewidth=2, label='T2 val seg dice loss')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Seg T2 Dice Loss')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "epoch_T2_seg_dice_loss.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.ce_losses_classi, 'red', linewidth=2, label='train classi ce loss')
        plt.plot(iters, self.val_ce_losses_classi, 'coral', linewidth=2, label='val classi ce loss')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Classi Ce Loss')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "epoch_classi_ce_loss.png"))

        plt.cla()
        plt.close("all")

    def append_dice(self, epoch, train_dice_T1,all_train_dice_T1,val_dice_T1,all_val_dice_T1,
                    train_dice_T2,all_train_dice_T2,val_dice_T2,all_val_dice_T2,train_acc,val_acc):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.train_dices_T1.append(train_dice_T1)
        self.train_dices1_T1.append(all_train_dice_T1[0])
        self.train_dices2_T1.append(all_train_dice_T1[1])
        self.train_dices3_T1.append(all_train_dice_T1[2])
        self.train_dices4_T1.append(all_train_dice_T1[3])
        self.train_dices5_T1.append(all_train_dice_T1[4])
        self.train_dices6_T1.append(all_train_dice_T1[5])

        self.val_dices_T1.append(val_dice_T1)
        self.val_dices1_T1.append(all_val_dice_T1[0])
        self.val_dices2_T1.append(all_val_dice_T1[1])
        self.val_dices3_T1.append(all_val_dice_T1[2])
        self.val_dices4_T1.append(all_val_dice_T1[3])
        self.val_dices5_T1.append(all_val_dice_T1[4])
        self.val_dices6_T1.append(all_val_dice_T1[5])

        self.train_dices_T2.append(train_dice_T2)
        self.train_dices1_T2.append(all_train_dice_T2[0])
        self.train_dices2_T2.append(all_train_dice_T2[1])

        self.val_dices_T2.append(val_dice_T2)
        self.val_dices1_T2.append(all_val_dice_T2[0])
        self.val_dices2_T2.append(all_val_dice_T2[1])

        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)

        with open(os.path.join(self.log_dir, "epoch_train_dice_mean_T1.txt"), 'a') as f:
            f.write(str(train_dice_T1))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_1_T1.txt"), 'a') as f:
            f.write(str(all_train_dice_T1[0]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_2_T1.txt"), 'a') as f:
            f.write(str(all_train_dice_T1[1]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_3_T1.txt"), 'a') as f:
            f.write(str(all_train_dice_T1[2]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_4_T1.txt"), 'a') as f:
            f.write(str(all_train_dice_T1[3]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_5_T1.txt"), 'a') as f:
            f.write(str(all_train_dice_T1[4]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_6_T1.txt"), 'a') as f:
            f.write(str(all_train_dice_T1[5]))
            f.write("\n")


        with open(os.path.join(self.log_dir, "epoch_val_dice_mean_T1.txt"), 'a') as f:
            f.write(str(val_dice_T1))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_1_T1.txt"), 'a') as f:
            f.write(str(all_val_dice_T1[0]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_2_T1.txt"), 'a') as f:
            f.write(str(all_val_dice_T1[1]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_3_T1.txt"), 'a') as f:
            f.write(str(all_val_dice_T1[2]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_4_T1.txt"), 'a') as f:
            f.write(str(all_val_dice_T1[3]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_5_T1.txt"), 'a') as f:
            f.write(str(all_val_dice_T1[4]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_6_T1.txt"), 'a') as f:
            f.write(str(all_val_dice_T1[5]))
            f.write("\n")

        with open(os.path.join(self.log_dir, "epoch_train_dice_mean_T2.txt"), 'a') as f:
            f.write(str(train_dice_T2))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_1_T2.txt"), 'a') as f:
            f.write(str(all_train_dice_T2[0]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_2_T2.txt"), 'a') as f:
            f.write(str(all_train_dice_T2[1]))
            f.write("\n")


        with open(os.path.join(self.log_dir, "epoch_val_dice_mean_T2.txt"), 'a') as f:
            f.write(str(val_dice_T2))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_1_T2.txt"), 'a') as f:
            f.write(str(all_val_dice_T2[0]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_2_T2.txt"), 'a') as f:
            f.write(str(all_val_dice_T2[1]))
            f.write("\n")

        with open(os.path.join(self.log_dir, "epoch_train_acc.txt"), 'a') as f:
            f.write(str(train_acc))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_acc.txt"), 'a') as f:
            f.write(str(val_acc))
            f.write("\n")


        self.writer.add_scalar('train_dice_T1', train_dice_T1, epoch)
        self.writer.add_scalar('val_dice_T1', val_dice_T1, epoch)
        self.writer.add_scalar('train_dice_T2', train_dice_T2, epoch)
        self.writer.add_scalar('val_dice_T2', val_dice_T2, epoch)
        self.writer.add_scalar('train_acc', train_acc, epoch)
        self.writer.add_scalar('val_acc', val_acc, epoch)
        # self.writer.add_scalar('all_train_dice', all_train_dice, epoch)
        # self.writer.add_scalar('all_val_loss', all_val_dice, epoch)
        self.dice_plot()

    def dice_plot(self):
        iters = range(len(self.train_dices_T1))

        plt.figure()
        plt.plot(iters, self.train_dices_T1, 'red', linewidth=2, label='T1 mean train dice')
        plt.plot(iters, self.val_dices_T1, 'coral', linewidth = 2, label='T1 mean val dice')


        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('T1 Mean Dice')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "T1_mean_epoch_dice.png"))

        plt.cla()
        plt.close("all")



        plt.figure()
        plt.plot(iters, self.train_dices1_T1, 'red', linewidth=2, label='T1 train dice 1')
        plt.plot(iters, self.val_dices1_T1, 'coral', linewidth = 2, label='T1 val dice 1')


        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('T1 Dice 1')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "T1_epoch_dice_1.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.train_dices2_T1, 'red', linewidth=2, label='T1 train dice 2')
        plt.plot(iters, self.val_dices2_T1, 'coral', linewidth=2, label='T1 val dice 2')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('T1 Dice 2')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "T1_epoch_dice_2.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.train_dices3_T1, 'red', linewidth=2, label='T1 train dice 3')
        plt.plot(iters, self.val_dices3_T1, 'coral', linewidth=2, label='T1 val dice 3')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('T1 Dice 3')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "T1_epoch_dice_3.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.train_dices4_T1, 'red', linewidth=2, label='T1 train dice 4')
        plt.plot(iters, self.val_dices4_T1, 'coral', linewidth=2, label='T1 val dice 4')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('T1 Dice 4')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "T1_epoch_dice_4.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.train_dices5_T1, 'red', linewidth=2, label='T1 train dice 5')
        plt.plot(iters, self.val_dices5_T1, 'coral', linewidth=2, label='T1 val dice 5')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('T1 Dice 5')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "T1_epoch_dice_5.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.train_dices6_T1, 'red', linewidth=2, label='T1 train dice 6')
        plt.plot(iters, self.val_dices6_T1, 'coral', linewidth=2, label='T1 val dice 6')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('T1 Dice 6')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "T1_epoch_dice_6.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.train_dices_T2, 'red', linewidth=2, label='T2 mean train dice')
        plt.plot(iters, self.val_dices_T2, 'coral', linewidth=2, label='T2 mean val dice')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('T2 Mean Dice')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "T2_mean_epoch_dice.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.train_dices1_T2, 'red', linewidth=2, label='T2 train dice 1')
        plt.plot(iters, self.val_dices1_T2, 'coral', linewidth=2, label='T2 val dice 1')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('T2 Dice 1')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "T2_epoch_dice_1.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.train_dices2_T2, 'red', linewidth=2, label='T2 train dice 2')
        plt.plot(iters, self.val_dices2_T2, 'coral', linewidth=2, label='T2 val dice 2')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('T2 Dice 2')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "T2_epoch_dice_2.png"))

        plt.cla()
        plt.close("all")


        plt.figure()
        plt.plot(iters, self.train_accs, 'red', linewidth=2, label='train acc')
        plt.plot(iters, self.val_accs, 'coral', linewidth = 2, label='val acc')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "acc.png"))

        plt.cla()
        plt.close("all")