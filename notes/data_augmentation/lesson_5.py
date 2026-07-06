import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from notes.data_augmentation.lesson_3 import Food3Model
from notes.data_augmentation.lesson_2 import class_names

def make_prediction(image_path: str, model: torch.nn.Module, label: str, label_names: list[str], transform: transforms.Compose | None) -> str:
    """
    Uses the given *model* to determine/classify the given image at *image_path*, comparing it to *label_names*.
    Workflow:
     - Reads *image_path* with torchvision.io.read_image and normalizes pixel values to [0, 1]
     - Passes data through *transform* if there is one
     - Passes data through *model*
     - Calls argmax on model logits to determine most likely result
     - Plots original image with its *label*, and predicted label

    :param image_path: Path of where the image is located. Expects a **str**
    :param model: Model that should be used to make predictions on what the image is
    :param label: The true image label
    :param label_names: The possible labels the model outputs
    :param transform: A transforms.Compose that the image is passed through before evaluating
    :return: Predicted label of image
    """

    imported_image = torchvision.io.read_image(image_path)
    image = imported_image / 255
    if transform:
        image = transform(image)

    model.eval()
    with torch.inference_mode():
        y_pred = torch.argmax(model(image.unsqueeze(0)), dim=1)

    y = label_names[y_pred]

    plt.imshow(imported_image.permute(1, 2, 0))
    plt.title(f"Label of image: {label} | Predicted label: {y}")
    plt.axis(False)
    plt.show()

    return y

transform = transforms.Compose([
    transforms.Resize((64, 64))
])

model = Food3Model(3, 10, 3)
model.load_state_dict(torch.load(f="models_state_dict/Food3_with_trivial_augment_CNN.pth"))

make_prediction(image_path="data/slice_of_pizza.png",
                model=model,
                label="pizza",
                label_names=class_names,
                transform=transform)