from ipywidgets import IntProgress, Layout
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
        description= f'{self.step_description} Movies: 0/{end_amount} movie(s)',
        layout=Layout(width='45%')
        )
            
        # Display the progress bar
        display(self.progress_bar)
    
    def update_progress_bar(self, increament):
        self.progress_bar.value += increament
        if self.progress_bar.value == self.progress_bar.max:
            self.progress_bar.description = f'{self.step_description} Movies: {self.progress_bar.value}/{self.progress_bar.max} Complete!'
        else:
            self.progress_bar.description = f'{self.step_description} Movies: {self.progress_bar.value}/{self.progress_bar.max} movie(s)'