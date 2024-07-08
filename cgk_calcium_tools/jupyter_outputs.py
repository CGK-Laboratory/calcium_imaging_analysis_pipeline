from ipywidgets import IntProgress, Layout, HBox, Label
from IPython.display import display

class progress_bar():
    def __init__(self, end_amount, step_description):
        self.step_description = step_description

        # Initialize progress bar
        self.progress_bar = IntProgress(
        value=0, 
        min=0,
        max=end_amount, 
        style={
            'bar_color': '#3385ff',
            'description_width': 'auto'
            },
        layout=Layout(width='370px')
        )

        self.description_label = Label(value = f'{self.step_description} Movies: 0/{end_amount} movie(s)')

        self.progress_box = HBox([self.description_label, self.progress_bar])

        # Display the progress bar
        display(self.progress_box)
    
    def update_progress_bar(self, increment):
        self.progress_bar.value += increment
        if self.progress_bar.value == self.progress_bar.max:
            self.description_label = f'{self.step_description} Movies: {self.progress_bar.value}/{self.progress_bar.max} Complete!'
        else:
            self.description_label = f'{self.step_description} Movies: {self.progress_bar.value}/{self.progress_bar.max} movie(s)'