import streamlit as st
import pandas as pd
import numpy as np

st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')


@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)

# import normalization as norm
# import tkinter as tk
# from pprint import pprint as pp
#
#
# entry_data = {}
#
#
# def save_input(category: tk.Entry, data: tk.Text) -> None:
#
#     new_category = category.get()
#
#     if new_category == '':
#         return
#
#     entry_data[new_category] = []
#
#     new_data = data.get("1.0", tk.END)
#     new_data = new_data.split('\n')
#
#     for i in new_data:
#         if i != tk.END:
#             try:
#                 entry_data[new_category].append(float(i))
#             except ValueError:
#                 continue
#
#     print_result("Saved!")
#
#
# def clear_input(category: tk.Entry, data: tk.Text) -> None:
#
#     category.delete(0, tk.END)
#     data.delete("1.0", tk.END)
#
#
# def restart(category: tk.Entry, data: tk.Text):
#
#     clear_input(category, data)
#     entry_data.clear()
#
#
# def print_result(text: str):
#
#     output = tk.Label(master=frame_data_output, text=text, width=40)
#     output.grid(row=1, column=0, padx=10, pady=10)
#
#
# def add_frame_data_input() -> (tk.Entry, tk.Text):
#
#     def add_frame_category_entry() -> tk.Entry:
#
#         frame_category_input = tk.Frame(master=frame_data_input)
#         frame_category_input.grid(row=0, column=0, padx=10, pady=10)
#
#         label_data_category = tk.Label(master=frame_category_input, text="Category", width=10)
#         entry_data_category = tk.Entry(master=frame_category_input, width=30)
#
#         label_data_category.grid(row=0, column=0, padx=10, pady=10)
#         entry_data_category.grid(row=0, column=1, padx=10, pady=10)
#
#         return entry_data_category
#
#     def add_frame_data_entry() -> tk.Text:
#
#         frame_text_input = tk.Frame(master=frame_data_input)
#         frame_text_input.grid(row=1, column=0, padx=10, pady=10)
#
#         label_data_entry = tk.Label(master=frame_text_input, text="Data", width=10)
#         text_data_entry = tk.Text(master=frame_text_input, width=23, height=10)
#
#         label_data_entry.grid(row=0, column=0, padx=10, pady=10)
#         text_data_entry.grid(row=0, column=1, padx=10, pady=10)
#
#         return text_data_entry
#
#     cat_entry = add_frame_category_entry()
#     data_entry = add_frame_data_entry()
#
#     def add_frame_input_ops() -> None:
#
#         frame_input_ops = tk.Frame(master=frame_data_input)
#         frame_input_ops.grid(row=1, column=1, padx=10, pady=10)
#
#         button_save_entry = tk.Button(master=frame_input_ops, text="Save Fields", width=10, command=lambda: save_input(cat_entry, data_entry))
#         button_clear_entry = tk.Button(master=frame_input_ops, text="Clear Fields", width=10, command=lambda: clear_input(cat_entry, data_entry))
#         button_restart = tk.Button(master=frame_input_ops, text="Restart", width=10, command=lambda: restart(cat_entry, data_entry))
#
#         button_save_entry.grid(row=0, column=0, padx=10, pady=10)
#         button_clear_entry.grid(row=1, column=0, padx=10, pady=10)
#         button_restart.grid(row=2, column=0, padx=10, pady=10)
#
#     add_frame_input_ops()
#
#
# def add_frame_data_output() -> None:
#
#     label_data_category = tk.Label(master=frame_data_output, text="Results", width=40)
#     label_data_category.grid(row=0, column=0, pady=10)
#
#
# def add_frame_calc_ops() -> None:
#
#     button_width = 25
#
#     label_std_ops = tk.Label(master=frame_calc_ops, width=button_width, text="Standard Normalization")
#     label_fs_ops = tk.Label(master=frame_calc_ops, width=button_width, text="Feature Scaling")
#
#     label_std_ops.grid(row=1, column=0, padx=10, pady=10)
#     label_fs_ops.grid(row=1, column=1, padx=10, pady=10)
#
#     button_std_dev = tk.Button(master=frame_calc_ops, width=button_width, text="Show Standard Deviation", command=lambda: print_result(
#         norm.std_dev(entry_data)))
#     button_std_mean = tk.Button(master=frame_calc_ops, width=button_width, text="Show Mean", command=lambda: print_result(
#         norm.std_mean(entry_data)))
#     button_std_norm = tk.Button(master=frame_calc_ops, width=button_width, text="Show Standard Normalized Data", command=lambda: print_result(
#         norm.std_norm(entry_data)))
#     button_fs_min = tk.Button(master=frame_calc_ops, width=button_width, text="Show Min", command=lambda: print_result(
#         norm.fs_min(entry_data)))
#     button_fs_max = tk.Button(master=frame_calc_ops, width=button_width, text="Show Max", command=lambda: print_result(
#         norm.fs_max(entry_data)))
#     button_fs_scaled = tk.Button(master=frame_calc_ops, width=button_width, text="Show Feature Scaled Data", command=lambda: print_result(
#         norm.fs_norm(entry_data)))
#
#     button_std_mean.grid(row=2, column=0, padx=10, pady=10)
#     button_std_dev.grid(row=3, column=0, padx=10, pady=10)
#     button_std_norm.grid(row=4, column=0, padx=10, pady=10)
#     button_fs_min.grid(row=2, column=1, padx=10, pady=10)
#     button_fs_max.grid(row=3, column=1, padx=10, pady=10)
#     button_fs_scaled.grid(row=4, column=1, padx=10, pady=10)
#
#
# if __name__ == "__main__":
#     window = tk.Tk()
#     window.title("Normalization App")
#     window.rowconfigure(0, minsize=50, weight=1)
#     window.columnconfigure(0, minsize=200, weight=1)
#
#     window.update()
#
#     frame_data_input = tk.Frame(master=window, relief=tk.GROOVE, borderwidth=1)
#     frame_data_input.grid(row=0, column=0, sticky='nsew')
#
#     frame_calc_ops = tk.Frame(master=window, relief=tk.GROOVE, borderwidth=1)
#     frame_calc_ops.grid(row=1, column=0, sticky='nsew')
#
#     frame_data_output = tk.Frame(master=window, relief=tk.GROOVE, borderwidth=1)
#     frame_data_output.grid(row=0, rowspan=2, column=1, sticky='nsew')
#
#     add_frame_data_input()
#     add_frame_calc_ops()
#     add_frame_data_output()
#     print(window.winfo_width())
#     window.mainloop()
#
#
