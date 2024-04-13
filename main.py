from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import random
import numpy as np

class train:

    def __init__(self):
        self.train_losses = []
        self.train_acc    = []
        self.train_loss_per_epoc = []
        self.train_acc_per_epoc = []
        

    # Training
    def run(self,net, device, trainloader, optimizer, scheduler, criterion,epoch):
        lrs = []
        #print('Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        #total = 0
        processed = 0
        pbar = tqdm(trainloader)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            # get samples
            inputs, targets = inputs.to(device), targets.to(device)

            # Init
            optimizer.zero_grad()

            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            outputs = net(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)
            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # appending the learning rate for every batch
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

            train_loss += loss.item()
            
            _, predicted = outputs.max(1)
            processed += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(desc= f'Epoch: {epoch},Loss={loss.item():3.2f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
            
        self.train_loss_per_epoc.append(sum(self.train_losses[len(self.train_losses)-len(trainloader):])/len(trainloader))
        self.train_acc_per_epoc.append(sum(self.train_acc[len(self.train_acc) - len(trainloader):])/len(trainloader))
    
        #return lrs
    def plot_trining_process(self):
      fig, axs = plt.subplots(1,2,figsize=(15,7))
      axs[0].plot([float(loss.cpu()) for loss in self.train_loss_per_epoc])
      axs[0].set_title("Training Loss")
      axs[1].plot(self.train_acc_per_epoc)
      axs[1].set_title("Training Accuracy")

class test:
  
    def __init__(self):

        self.test_losses = []
        self.test_acc    = []
        self.wrongly_predicted_samples = None
        self.classes = ('plane', 'car', 'bird', 'cat',
          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.mean = np.array((0.4914, 0.4822, 0.4465))
        self.std = np.array((0.2023, 0.1994, 0.2010))

    def execute(self, model, device, test_loader, criterion):
      wrongly_predicted_samples = {}
      model.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data.to(device), target.to(device)
              output = model(data)
              test_loss += criterion(output, target).item()  # sum up batch loss
              pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              correct += pred.eq(target.view_as(pred)).sum().item()

              # storing the misclassified samples
              misclassified_index = torch.nonzero(target != pred.squeeze(1)).squeeze()
              if wrongly_predicted_samples == {}:
                wrongly_predicted_samples = {
                    "image" : data[misclassified_index],
                    "target" : target[misclassified_index],
                    "pred" : pred.squeeze(1)[misclassified_index]
                }
              else:
                wrongly_predicted_samples["image"] = torch.cat([wrongly_predicted_samples["image"], data[misclassified_index]], dim = 0)
                wrongly_predicted_samples["target"] = torch.cat([wrongly_predicted_samples["target"], target[misclassified_index]], dim = 0)
                wrongly_predicted_samples["pred"] = torch.cat([wrongly_predicted_samples["pred"], pred.squeeze(1)[misclassified_index] ], dim = 0)

      test_loss /= len(test_loader.dataset)
      self.test_losses.append(test_loss)

      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))

      self.test_acc.append(100. * correct / len(test_loader.dataset))
      self.wrongly_predicted_samples = wrongly_predicted_samples
      return wrongly_predicted_samples

    def get_misclassified_samples(self):
      return self.wrongly_predicted_samples

    def plt_misclassified_images(self, grid_shape = (2, 5)):
      samples = random.sample(range(0,self.wrongly_predicted_samples["image"].shape[0]), grid_shape[0]*grid_shape[1])
      misclassified_images = self.wrongly_predicted_samples["image"][samples].cpu().numpy()
      predictions = [self.classes[category] for category in self.wrongly_predicted_samples["pred"][samples].cpu().tolist()]
      targets = [self.classes[category] for category in self.wrongly_predicted_samples["target"][samples].cpu().tolist()]

      # functions to show an misclassified image with label
      def imshow_misclassified(ax, img, pred, original):
          # unnormalize
          
          img = img * self.std[:, None, None] + self.mean[:, None, None]
          ax.imshow(np.transpose(img, (1, 2, 0)))
          ax.set_title({"target : {} -> pred : {}".format(original, pred)})

      fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(15, 6))
      axes = axes.flatten()
      # Display images
      for ax, image, pred, target in zip(axes, misclassified_images, predictions, targets):
        imshow_misclassified(ax, image, pred, target)

      plt.tight_layout()
      plt.show()

    
    def plot_test_process(self):
      fig, axs = plt.subplots(1,2,figsize=(15,7))
      axs[0].plot(self.test_losses)
      axs[0].set_title("Test Loss")
      axs[1].plot(self.test_acc)
      axs[1].set_title("Test Accuracy")
          
          
        
        
