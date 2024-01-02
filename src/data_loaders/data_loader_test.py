import numpy as np
import matplotlib.pyplot as plt

def plot_band(data, title, cmap=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.show()

def main():
    # Load Sentinel-2 data for segment_1
    data_path = '../data/experiments/segment_1/sentinel_data.npy'  # Update with your actual file path
    data = np.load(data_path)

    # Check the shape of the data
    print(f"Data shape: {data.shape}")  # Should be in the form (bands, height, width)

    # Plotting the first three bands as an RGB image
    # Assuming the first three bands can be combined into an RGB image
    if data.shape[0] >= 3:
        rgb = np.dstack((data[2], data[1], data[0]))  # Adjust the indices based on the bands
        plt.imshow(rgb)
        plt.title("RGB Composite")
        plt.show()
    else:
        print("Not enough bands for RGB composite")

    # Optionally, plot individual bands
    # Plot the first band as an example
    plot_band(data[0], "Band 1 (Example)", cmap='gray')

if __name__ == "__main__":
    main()
