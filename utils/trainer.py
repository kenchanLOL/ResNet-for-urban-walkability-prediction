from tqdm import tqdm
import time
import torch

NUM_EPOCHS = 50
EVAL_PER_EPOCH = 5
GRADIENT_ACCUMULATION_STEPS = 8 #update as a batch of 32 (4*8)

class Trainer():
    def __init__(self, save_model_prefix, optimizer, scheduler, writer, train_loader, test_loader, val_loader, device, eval_per_epoch = None, gradient_accumulation = None, **kwargs):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.score_loss = torch.nn.MSELoss()
        self.writer = writer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.eval_per_epoch = eval_per_epoch if eval_per_epoch else EVAL_PER_EPOCH
        self.gradient_accumulation = gradient_accumulation if gradient_accumulation else GRADIENT_ACCUMULATION_STEPS
        self.device = device
        self.save_model_prefix = save_model_prefix

    def train(self, model, num_epoch = NUM_EPOCHS):
        self.num_epoch = num_epoch
        epoch_pbar = tqdm(range(num_epoch), desc='Epochs', position=0)
        for epoch in epoch_pbar:
            train_loss, train_acc, total_sample, time = self.train_epoch(epoch, model)
            # Evaluate
            best_acc = 0
            if epoch % EVAL_PER_EPOCH == 0 or epoch == len(self.train_loader)-1:
                s_val_acc, total_val_sample = self.evaluate(model, epoch, logging = True)
                if s_val_acc/total_val_sample >  best_acc:
                    best_acc = s_val_acc/total_val_sample
                    torch.save(model.state_dict(), f"{self.save_model_prefix}_{int(best_acc*1000)}.pth")
                    print(f"Best Model Updated at epoch {epoch+1} with acc = {best_acc}")
                else:
                    print(f"Model Not Updated at epoch {epoch+1}")
            # epoch_pbar.set_postfix({'G_Acc': g_val_acc/total_val_sample, 'P_Acc': p_val_acc/total_val_sample, 'S_Acc': s_val_acc/total_val_sample})
            epoch_pbar.set_postfix({'S_Acc': s_val_acc/total_sample})
            # writer.add_scalar('Accuracy/Train_Global', g_train_acc/total_sample, epoch )
            # writer.add_scalar('Accuracy/Train_Patch', p_train_acc/total_sample, epoch )
            self.writer.add_scalar('Accuracy/Train_Score', train_acc/total_sample, epoch )
            self.writer.add_scalar('Metadata/Epoch_Duration(min)', time, epoch)
            self.writer.add_scalar('Metadata/Learning Rate', self.scheduler.get_last_lr()[0], epoch)
            return model

    def train_epoch(self, epoch, model):
        total_g_loss, total_p_loss, total_s_loss, total_t_loss = 0, 0, 0, 0
        g_train_acc, p_train_acc, s_train_acc = 0, 0, 0
        num_batches = len(self.train_loader)
        total_sample = 0
        start = time.time()
        step_pbar = tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.num_epoch}', position=0)
        # for idx, (left, right, labels) in enumerate(self.train_loader):
        for idx, (left, right, labels) in enumerate(self.train_loader):
            left[0] = left[0].to(self.device)
            left[1] = left[1].to(self.device)
            right[0] = right[0].to(self.device)
            right[1] = right[1].to(self.device)
            labels = labels.to(self.device)
            y_left, y_right, y_score = model(left, right)
            # g_loss = global_loss(y_global, labels)
            # p_loss = patch_loss(y_patch, labels)
            y_score = y_score.squeeze()
            s_loss = self.score_loss(y_score, labels.float())
            total_loss = s_loss
            total_loss.backward()
            if (idx+1) % GRADIENT_ACCUMULATION_STEPS == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                # break
            # logging
            ### Loss
            # total_g_loss += g_loss.item()
            # total_p_loss += p_loss.item()
            total_s_loss += s_loss.item()
            total_t_loss += total_loss.item()
            ### Acc
            # _, y_global = torch.max(y_global, 1)
            # _, y_patch = torch.max(y_patch, 1)
            y_score = torch.where(y_score > 0.5, 1.0, 0.0).squeeze()
            # g_train_acc += (labels == y_global).sum().item()
            # p_train_acc += (labels == y_patch).sum().item()
            s_train_acc += (labels == y_score).sum().item()
            total_sample += labels.shape[0]
            step_pbar.update(1)
            # step_pbar.set_postfix({'Global Loss': total_g_loss/(idx+1), 'Patch Loss': total_p_loss/(idx+1), 'Score Loss': total_s_loss/(idx+1), 'Total Loss':total_t_loss/(idx+1), 'Global Acc':g_train_acc/total_sample, 'Patch Acc':p_train_acc/total_sample, 'Score Acc':s_train_acc/total_sample})
            step_pbar.set_postfix({'Score Loss': total_s_loss/(idx+1), 'Score Acc':s_train_acc/total_sample})
            if idx % 300 == 0 and idx != 0:
                # writer.add_scalar('Loss/Global', total_g_loss/(idx+1), epoch*(num_batches)+idx)
                # writer.add_scalar('Loss/Patch', total_p_loss/(idx+1), epoch*(num_batches)+idx)
                self.writer.add_scalar('Loss/Score', total_s_loss/(idx+1), epoch*(num_batches)+idx)
                # writer.add_scalar('Loss/Total', total_t_loss/(idx+1), epoch*(num_batches)+idx)

        step_pbar.close()
        self.scheduler.step()
        end = time.time()
        duration = (end - start)/60
        return total_t_loss, s_train_acc, total_sample, duration

    def evaluate(self, model, epoch = None, logging = False, val_df = None):
        acc = 0
        total_sample = 0
        pbar = tqdm(total=len(self.val_loader), position=0)
        model.eval()
        sigmoid = torch.nn.Sigmoid()
        with torch.no_grad():
            for idx, (left, right, labels, df_index) in enumerate(self.val_loader):
                left[0] = left[0].to(self.device)
                left[1] = left[1].to(self.device)
                right[0] = right[0].to(self.device)
                right[1] = right[1].to(self.device)
                labels = labels.to(self.device)
                score_left = model(left)
                score_right = model(right)
                y_score = score_right - score_left
                y_score = sigmoid(y_score)
                y_score = torch.where(y_score > 0.5, 1.0, 0.0).squeeze()

                acc += (y_score == labels).sum().item()
                total_sample += labels.shape[0]

                pbar.update(1)
                pbar.set_postfix({'acc': acc/total_sample})
                if val_df is not None:
                    val_df.loc[df_index, "prediction_left"] = score_left.cpu().numpy()
                    val_df.loc[df_index,"prediction_right"] = score_right.cpu().numpy()

                if logging:
                    # writer.add_scalar('Accuracy/Val_Global', g_val_acc/total_val_sample, epoch )
                    # writer.add_scalar('Accuracy/Val_Patch', p_val_acc/total_val_sample, epoch )
                    self.writer.add_scalar('Accuracy/Val_Score', acc/total_sample, epoch )
                # break
        pbar.close()
        return acc, total_sample
