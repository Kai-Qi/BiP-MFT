import os
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import scipy

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.ce_losses= []
        self.dice_losses= []
        self.val_losses= []
        self.val_dice_losses= []
        self.val_ce_losses= []

        self.train_dices=[]
        self.train_dices1 = []
        self.train_dices2 = []
        self.train_dices3 = []
        self.train_dices4 = []
        self.train_dices5 = []
        self.train_dices6 = []

        self.mean_dice=[]
        self.val_dices=[]
        self.val_dices1 = []
        self.val_dices2 = []
        self.val_dices3 = []
        self.val_dices4 = []
        self.val_dices5 = []
        self.val_dices6 = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, ce_loss,dice_loss,val_loss,val_dice_loss,val_ce_loss):
    # def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.ce_losses.append(ce_loss)
        self.dice_losses.append(dice_loss)
        self.val_losses.append(val_loss)
        self.val_dice_losses.append(val_dice_loss)
        self.val_ce_losses.append(val_ce_loss)


        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_ce_loss.txt"), 'a') as f:
            f.write(str(ce_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_ce_loss.txt"), 'a') as f:
            f.write(str(val_ce_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_dice_loss.txt"), 'a') as f:
            f.write(str(dice_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_loss.txt"), 'a') as f:
            f.write(str(val_dice_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.writer.add_scalar('ce_loss', ce_loss, epoch)
        self.writer.add_scalar('val_ce_loss', val_ce_loss, epoch)
        self.writer.add_scalar('dice_loss', dice_loss, epoch)
        self.writer.add_scalar('val_dice_loss', val_dice_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_losses, 'coral', linewidth = 2, label='val loss')

        # try:
        #     if len(self.losses) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
        #              label='smooth train loss')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_losses, num, 3), '#8B4513', linestyle='--', linewidth=2,
        #              label='smooth val loss')
        #
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.dice_losses, 'red', linewidth=2, label='train dice loss')
        plt.plot(iters, self.val_dice_losses, 'coral', linewidth=2, label='val dice loss')

        # try:
        #     if len(self.losses) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.dice_losses, num, 3), 'green', linestyle='--', linewidth=2,
        #              label='smooth train dice loss')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_dice_losses, num, 3), '#8B4513', linestyle='--', linewidth=2,
        #              label='smooth val dice loss')
        #
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Dice Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_dice_loss.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.ce_losses, 'red', linewidth=2, label='train ce loss')
        plt.plot(iters, self.val_ce_losses, 'coral', linewidth=2, label='val ce loss')

        # try:
        #     if len(self.losses) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.ce_losses, num, 3), 'green', linestyle='--', linewidth=2,
        #              label='smooth train ce loss')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_ce_losses, num, 3), '#8B4513', linestyle='--',
        #              linewidth=2,
        #              label='smooth val ce loss')
        #
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Ce Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_ce_loss.png"))

        plt.cla()
        plt.close("all")

    def append_dice(self, epoch, train_dice,all_train_dice,val_dice,all_val_dice):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        mean_dice=(torch.sum(all_val_dice)-all_val_dice[0])/(len(all_val_dice)-1)

        self.train_dices.append(train_dice)
        self.train_dices1.append(all_train_dice[0])
        self.train_dices2.append(all_train_dice[1])
        self.train_dices3.append(all_train_dice[2])
        self.train_dices4.append(all_train_dice[3])
        self.train_dices5.append(all_train_dice[4])
        self.train_dices6.append(all_train_dice[5])


        self.val_dices.append(val_dice)
        self.val_dices1.append(all_val_dice[0])
        self.val_dices2.append(all_val_dice[1])
        self.val_dices3.append(all_val_dice[2])
        self.val_dices4.append(all_val_dice[3])
        self.val_dices5.append(all_val_dice[4])
        self.val_dices6.append(all_val_dice[5])
        self.mean_dice.append(mean_dice)

        with open(os.path.join(self.log_dir, "epoch_train_dice_mean.txt"), 'a') as f:
            f.write(str(train_dice))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_1.txt"), 'a') as f:
            f.write(str(all_train_dice[0]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_2.txt"), 'a') as f:
            f.write(str(all_train_dice[1]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_3.txt"), 'a') as f:
            f.write(str(all_train_dice[2]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_4.txt"), 'a') as f:
            f.write(str(all_train_dice[3]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_5.txt"), 'a') as f:
            f.write(str(all_train_dice[4]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_dice_6.txt"), 'a') as f:
            f.write(str(all_train_dice[5]))
            f.write("\n")

        with open(os.path.join(self.log_dir, "epoch_val_dice_mean.txt"), 'a') as f:
            f.write(str(val_dice))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_1.txt"), 'a') as f:
            f.write(str(all_val_dice[0]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_2.txt"), 'a') as f:
            f.write(str(all_val_dice[1]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_3.txt"), 'a') as f:
            f.write(str(all_val_dice[2]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_4.txt"), 'a') as f:
            f.write(str(all_val_dice[3]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_5.txt"), 'a') as f:
            f.write(str(all_val_dice[4]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_dice_6.txt"), 'a') as f:
            f.write(str(all_val_dice[5]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_mean_dice.txt"), 'a') as f:
            f.write(str(mean_dice))
            f.write("\n")


        self.writer.add_scalar('train_dice', train_dice, epoch)
        self.writer.add_scalar('val_dice', val_dice, epoch)
        self.writer.add_scalar('mean_dice', mean_dice, epoch)
        # self.writer.add_scalar('all_train_dice', all_train_dice, epoch)
        # self.writer.add_scalar('all_val_loss', all_val_dice, epoch)
        self.dice_plot()

    def dice_plot(self):
        iters = range(len(self.train_dices))

        plt.figure()
        plt.plot(iters, self.train_dices, 'red', linewidth=2, label='mean train dice')
        plt.plot(iters, self.val_dices, 'coral', linewidth = 2, label='mean val dice')

        # try:
        #     if len(self.train_dices) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.train_dices, num, 3), 'green', linestyle='--', linewidth=2,
        #              label='smooth mean train dice')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_dices, num, 3), '#8B4513', linestyle='--', linewidth=2,
        #              label='smooth mean val dice')
        #
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Dice')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "mean_epoch_dice.png"))

        plt.cla()
        plt.close("all")




        plt.figure()
        plt.plot(iters, self.train_dices1, 'red', linewidth=2, label='train dice 1')
        plt.plot(iters, self.val_dices1, 'coral', linewidth = 2, label='val dice 1')

        # try:
        #     if len(self.train_dices) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.train_dices1, num, 3), 'green', linestyle='--', linewidth=2,
        #              label='smooth train dice 1')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_dices1, num, 3), '#8B4513', linestyle='--', linewidth=2,
        #              label='smooth mean val dice 1')
        #
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Dice 1')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_dice_1.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.train_dices2, 'red', linewidth=2, label='train dice 2')
        plt.plot(iters, self.val_dices2, 'coral', linewidth=2, label='val dice 2')

        # try:
        #     if len(self.train_dices) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.train_dices2, num, 3), 'green', linestyle='--', linewidth=2,
        #              label='smooth train dice 2')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_dices2, num, 3), '#8B4513', linestyle='--', linewidth=2,
        #              label='smooth mean val dice 2')
        #
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Dice 2')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_dice_2.png"))

        plt.cla()
        plt.close("all")




        plt.figure()
        plt.plot(iters, self.train_dices3, 'red', linewidth=2, label='train dice 3')
        plt.plot(iters, self.val_dices3, 'coral', linewidth=2, label='val dice 3')

        # try:
        #     if len(self.train_dices) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.train_dices3, num, 3), 'green', linestyle='--', linewidth=2,
        #              label='smooth train dice 3')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_dices3, num, 3), '#8B4513', linestyle='--', linewidth=2,
        #              label='smooth mean val dice 3')
        #
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Dice 3')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_dice_3.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.train_dices4, 'red', linewidth=2, label='train dice 4')
        plt.plot(iters, self.val_dices4, 'coral', linewidth=2, label='val dice 4')

        # try:
        #     if len(self.train_dices) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.train_dices4, num, 3), 'green', linestyle='--', linewidth=2,
        #              label='smooth train dice 4')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_dices4, num, 3), '#8B4513', linestyle='--', linewidth=2,
        #              label='smooth mean val dice 4')
        #
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Dice 4')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_dice_4.png"))

        plt.cla()
        plt.close("all")


        plt.figure()
        plt.plot(iters, self.train_dices5, 'red', linewidth=2, label='train dice 5')
        plt.plot(iters, self.val_dices5, 'coral', linewidth=2, label='val dice 5')

        # try:
        #     if len(self.train_dices) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.train_dices5, num, 3), 'green', linestyle='--', linewidth=2,
        #              label='smooth train dice 5')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_dices5, num, 3), '#8B4513', linestyle='--', linewidth=2,
        #              label='smooth mean val dice 5')
        #
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Dice 5')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_dice_5.png"))

        plt.cla()
        plt.close("all")



        plt.figure()
        plt.plot(iters, self.train_dices6, 'red', linewidth=2, label='train dice 6')
        plt.plot(iters, self.val_dices6, 'coral', linewidth=2, label='val dice 6')

        # try:
        #     if len(self.train_dices) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.train_dices6, num, 3), 'green', linestyle='--', linewidth=2,
        #              label='smooth train dice 6')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_dices6, num, 3), '#8B4513', linestyle='--', linewidth=2,
        #              label='smooth mean val dice 6')
        #
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Dice 6')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_dice_6.png"))

        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.mean_dice, 'red', linewidth=2, label='mean dice')

        # try:
        #     if len(self.train_dices) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.mean_dice, num, 3), 'green', linestyle='--', linewidth=2,
        #              label='smooth mean dice')
        #
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Dice(execpt background)')
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.log_dir, "mean_dice.png"))

        plt.cla()
        plt.close("all")
