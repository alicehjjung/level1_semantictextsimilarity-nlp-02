import matplotlib.pyplot as plt


def view_scatter(predicted, actual):
    plt.scatter(predicted.cpu().numpy(), actual.cpu().numpy())

    plt.title('Predicted vs. Actual Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')

    plt.show()