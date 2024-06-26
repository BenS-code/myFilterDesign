import tkinter as tk
from tkinter import ttk, filedialog, END
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, cheby1, bessel, freqz, lfilter, tf2zpk
import pandas as pd


class FilterDesignApp:
    def __init__(self, master):
        self.master = master
        master.title("myFilterDesign")
        # Set minimum size of the window
        master.geometry('1600x800')
        master.minsize(1600, 800)

        # Left side widgets (sliders, buttons, comboboxes)
        self.left_frame = tk.Frame(master)
        self.left_frame.grid(row=0, column=0, padx=10, pady=5, sticky='nsew')

        # Signal type combobox
        self.signal_type_label = tk.Label(self.left_frame, text="Signal Type:")
        self.signal_type_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.signal_types = ['Sine', 'Step', 'Square', 'Triangle', 'Sawtooth', 'Imported File']
        self.signal_type_var = tk.StringVar()
        self.signal_type_combobox = ttk.Combobox(self.left_frame, textvariable=self.signal_type_var,
                                                 values=self.signal_types)
        self.signal_type_combobox.set('Step')  # Default signal type
        self.signal_type_combobox.grid(row=0, column=1, padx=10, pady=5, sticky='we')

        # Filter type combobox
        self.filter_type_label = tk.Label(self.left_frame, text="Filter Type:")
        self.filter_type_label.grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.filter_types = ['Butterworth', 'Bessel', 'Chebyshev']
        self.filter_type_var = tk.StringVar()
        self.filter_type_combobox = ttk.Combobox(self.left_frame, textvariable=self.filter_type_var,
                                                 values=self.filter_types)
        self.filter_type_combobox.set('Butterworth')  # Default filter type
        self.filter_type_combobox.grid(row=1, column=1, padx=10, pady=5, sticky='we')

        # Filter order slider
        self.order_label = tk.Label(self.left_frame, text="Filter Order:")
        self.order_label.grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.order_scale = tk.Scale(self.left_frame, from_=1, to=10, orient=tk.HORIZONTAL)
        self.order_scale.set(4)  # Default filter order
        self.order_scale.grid(row=2, column=1, padx=10, pady=5, sticky='we')

        # Cutoff frequency slider
        self.cutoff_label = tk.Label(self.left_frame, text="Cutoff Frequency:")
        self.cutoff_label.grid(row=3, column=0, padx=10, pady=5, sticky='w')
        self.cutoff_scale = tk.Scale(self.left_frame, from_=1, to=100, resolution=1, orient=tk.HORIZONTAL)
        self.cutoff_scale.set(18)  # Default cutoff frequency
        self.cutoff_scale.grid(row=3, column=1, padx=10, pady=5, sticky='we')

        # Sampling frequency slider
        self.sampling_label = tk.Label(self.left_frame, text="Sampling Frequency:")
        self.sampling_label.grid(row=4, column=0, padx=10, pady=5, sticky='w')
        self.sampling_scale = tk.Scale(self.left_frame, from_=1, to=200, orient=tk.HORIZONTAL)
        self.sampling_scale.set(100)  # Default sampling frequency
        self.sampling_scale.grid(row=4, column=1, padx=10, pady=5, sticky='we')

        # File upload button
        self.upload_button = tk.Button(self.left_frame, text="Upload CSV", command=self.upload_csv)
        self.upload_button.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky='we')

        # Select time column combobox
        self.time_col_label = tk.Label(self.left_frame, text="Time Column:")
        self.time_col_label.grid(row=6, column=0, padx=10, pady=5, sticky='w')
        self.time_col_var = tk.StringVar()
        self.time_col_combobox = ttk.Combobox(self.left_frame, textvariable=self.time_col_var, state='disabled')
        self.time_col_combobox.grid(row=6, column=1, padx=10, pady=5, sticky='we')

        # Select signal column combobox
        self.signal_col_label = tk.Label(self.left_frame, text="Signal Column:")
        self.signal_col_label.grid(row=7, column=0, padx=10, pady=5, sticky='w')
        self.signal_col_var = tk.StringVar()
        self.signal_col_combobox = ttk.Combobox(self.left_frame, textvariable=self.signal_col_var, state='disabled')
        self.signal_col_combobox.grid(row=7, column=1, padx=10, pady=5, sticky='we')

        self.filter_button = tk.Button(self.left_frame, text="Apply Filter", command=self.apply_filter)
        self.filter_button.grid(row=8, column=0, columnspan=2, padx=10, pady=5, sticky='we')

        # text entry for filter's coefficients
        self.filter_a_coefficients_label = tk.Label(self.left_frame, text="Filter Coefficients a =")
        self.filter_a_coefficients_label.grid(row=9, column=0, padx=10, pady=5, sticky='w')
        self.filter_a_coefficients_text = tk.Text(self.left_frame, height=6, width=20)
        self.filter_a_coefficients_text.grid(row=9, column=1, padx=10, pady=5, sticky='we')

        self.filter_b_coefficients_label = tk.Label(self.left_frame, text="Filter Coefficients b =")
        self.filter_b_coefficients_label.grid(row=10, column=0, padx=10, pady=5, sticky='w')
        self.filter_b_coefficients_text = tk.Text(self.left_frame, height=6, width=20)
        self.filter_b_coefficients_text.grid(row=10, column=1, padx=10, pady=5, sticky='we')

        self.close_button = tk.Button(self.left_frame, text="Close", command=master.quit)
        self.close_button.grid(row=11, column=0, columnspan=2, padx=10, pady=5, sticky='we')

        # Right side canvas
        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.grid(row=0, column=1, padx=10, pady=5, sticky='nsew')

        # Create a canvas to display the plot
        self.figure = plt.Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize DataFrame
        self.data = None

    def upload_csv(self):
        file_path = filedialog.askopenfilename(title="Select CSV File",
                                               filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if file_path:
            self.data = pd.read_csv(file_path)
            # Enable comboboxes and update choices
            self.time_col_combobox['state'] = 'readonly'
            self.signal_col_combobox['state'] = 'readonly'
            self.time_col_combobox['values'] = list(self.data.columns)
            self.signal_col_combobox['values'] = list(self.data.columns)
            # Add an option to signal types for the imported file
            self.signal_types = ['Sine', 'Step', 'Square', 'Triangle', 'Sawtooth', 'Imported File']
            self.signal_type_combobox['values'] = self.signal_types

    def generate_signal(self, signal_type, t):
        if signal_type == 'Sine':
            return np.sin(2 * np.pi * t)
        elif signal_type == 'Step':
            signal = np.zeros_like(t)
            signal[t >= 5] = 1
            return signal
        elif signal_type == 'Square':
            return np.sign(np.sin(2 * np.pi * t))
        elif signal_type == 'Triangle':
            return 2 * np.abs(2 * (t - np.floor(t + 0.5)))
        elif signal_type == 'Sawtooth':
            return np.linspace(-1, 1, len(t))
        elif signal_type == 'Imported File':
            if self.data is not None:
                return self.data[self.signal_col_var.get()].values
            else:
                return np.zeros(len(t))

    def apply_filter(self):
        # Generate signal
        sampling_freq = self.sampling_scale.get()
        nyquist = sampling_freq / 2
        t = np.linspace(0, 10, int(sampling_freq * 10))
        signal_type = self.signal_type_var.get()
        signal = self.generate_signal(signal_type, t)

        if signal_type == 'Imported File':
            t = self.data[self.time_col_var.get()].values

        # Get parameters from sliders and comboboxes
        filter_type = self.filter_type_var.get()
        order = int(self.order_scale.get())
        cutoff_freq = self.cutoff_scale.get() / nyquist  # Normalized cutoff frequency

        # Apply selected filter type
        a = 0
        b = 0
        if filter_type == 'Butterworth':
            [b, a] = butter(order, cutoff_freq, analog=False)
        elif filter_type == 'Bessel':
            [b, a] = bessel(order, cutoff_freq, analog=False)
        elif filter_type == 'Chebyshev':
            [b, a] = cheby1(order, 0.5, cutoff_freq, analog=False)

        filtered_signal = lfilter(b, a, signal)
        self.filter_a_coefficients_text.delete('1.0', END)
        self.filter_b_coefficients_text.delete('1.0', END)
        self.filter_a_coefficients_text.insert(END, a)
        self.filter_b_coefficients_text.insert(END, b)

        # Clear previous plot
        self.figure.clear()

        # Plot original and filtered signals
        ax1 = self.figure.add_subplot(212)
        ax1.plot(t, signal, label='Original Signal', alpha=0.7)
        ax1.plot(t, filtered_signal, label='Filtered Signal', alpha=0.7)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Original and Filtered Signals')
        ax1.legend()
        ax1.grid(True)

        # Compute frequency response of the filter
        w, h = freqz(b, a, worN=int(sampling_freq * 100))
        freq = 0.5 * 1 / np.pi * w * sampling_freq
        mag = np.abs(h)
        ax2 = self.figure.add_subplot(221)
        ax2.semilogx(freq, 20 * np.log10(mag), 'b')
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Magnitude [dB]')
        ax2.set_title(f'{filter_type} Frequency Response')
        ax2.set_ylim([-40, 10])
        ax2.grid(True)
        ax2.axvline(cutoff_freq * nyquist, color='r', linestyle='--',
                    label=f'Cutoff Frequency: {cutoff_freq * nyquist} Hz')
        ax2.legend()

        # Compute poles and zeros of the filter
        z, p, k = tf2zpk(b, a)
        ax3 = self.figure.add_subplot(222)
        ax3.plot(np.real(z), np.imag(z), 'bo', markersize=6, label='Zeros')
        ax3.plot(np.real(p), np.imag(p), 'rx', markersize=6, label='Poles')
        ax3.set_xlabel('Real')
        ax3.set_ylabel('Imaginary')
        ax3.set_title(f'{filter_type} Filter Poles and Zeros')
        ax3.grid(True)
        ax3.legend()

        # Draw canvas
        self.canvas.draw()


root = tk.Tk()
app = FilterDesignApp(root)
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.rowconfigure(0, weight=1)
root.mainloop()
