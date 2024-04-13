from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# from pl_bolts.datamodules import CIFAR10DataModule
import matplotlib.pyplot as plt
import random
from data_loader import de_normalize

def plt_misclassified_images(wrongly_predicted_samples, classes, grid_shape = (2, 5)):
      samples = random.sample(range(0,wrongly_predicted_samples["image"].shape[0]), grid_shape[0]*grid_shape[1])
      misclassified_images = wrongly_predicted_samples["image"][samples].cpu().numpy()
      predictions = [classes[category] for category in wrongly_predicted_samples["pred"][samples].cpu().tolist()]
      targets = [classes[category] for category in wrongly_predicted_samples["target"][samples].cpu().tolist()]

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



def plot_grad_cam_images(model, misclassified_data, classes, apply_grad = True):
    model.eval()
    target_layers = [model.layer3[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)
    # Plot the misclassified images
    fig = plt.figure(figsize=(12, 5))
    for i in range(10):
        sub = fig.add_subplot(2, 5, i+1)
        input_tensor = misclassified_data["image"][i].unsqueeze(dim=0)
        targets = [ClassifierOutputTarget(misclassified_data["target"][i])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = de_normalize(misclassified_data["image"][i].cpu())
        if apply_grad == True:
          visualization = show_cam_on_image(visualization, grayscale_cam, use_rgb=True, image_weight=0.7)

        # npimg = unnormalize(misclassified_images[i].cpu())
        # plt.imshow(npimg, cmap='gray', interpolation='none')
        # npimg = unnormalize(misclassified_images[i].cpu())

        plt.imshow(visualization)
        sub.set_title("Actual: {}, Pred: {}".format(classes[misclassified_data["target"][i]], classes[misclassified_data["pred"][i]]),color='red')
    plt.tight_layout()
    plt.show()