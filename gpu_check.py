import torch
import matplotlib.pyplot as plt
import time


def visualize_gpu_usage():
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    # Get the current device ID
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get total and allocated memory in bytes
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)

    # Calculate free memory
    free_memory = total_memory - allocated_memory

    # Convert to MB for readability
    total_memory_mb = total_memory / (1024 * 1024)
    allocated_memory_mb = allocated_memory / (1024 * 1024)
    cached_memory_mb = cached_memory / (1024 * 1024)
    free_memory_mb = free_memory / (1024 * 1024)

    # Print the stats
    print(f"Total Memory: {total_memory_mb:.2f} MB")
    print(f"Allocated Memory: {allocated_memory_mb:.2f} MB")
    print(f"Cached Memory: {cached_memory_mb:.2f} MB")
    print(f"Free Memory: {free_memory_mb:.2f} MB")

    # Data for plotting
    labels = ['Allocated', 'Cached', 'Free']
    sizes = [allocated_memory_mb, cached_memory_mb, free_memory_mb]
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    # Plotting the pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title('GPU Memory Usage')
    plt.show()

if __name__ == '__main__':
    # Run the function to visualize GPU usage
    while True:
        visualize_gpu_usage()
        time.sleep(5)

