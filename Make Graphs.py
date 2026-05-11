from graphviz import Digraph
import os


os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


# Create a new directed graph
dot = Digraph(comment = 'ANN Model', format = 'png')
dot.attr(rankdir = 'TB', size = '10')

# Input layer
dot.node('Input', 'Input\n(input_size)', shape = 'box', style = 'filled', color = 'lightgray')

# Hidden layers
layers = [
    ('Dense256', 'Dense\n256', 'yellow'),
    ('ReLU256', 'ReLU', 'pink'),
    ('Dropout256', 'Dropout\n0.2', 'lightblue'),
    ('Dense128', 'Dense\n128', 'yellow'),
    ('ReLU128', 'ReLU', 'pink'),
    ('Dropout128', 'Dropout\n0.2', 'lightblue'),
    ('Dense64', 'Dense\n64', 'yellow'),
    ('ReLU64', 'ReLU', 'pink'),
    ('Dropout64', 'Dropout\n0.2', 'lightblue'),
    ('Dense32', 'Dense\n32', 'yellow'),
    ('ReLU32', 'ReLU', 'pink'),
    ('Dropout32', 'Dropout\n0.2', 'lightblue'),
    ('Dense16', 'Dense\n16', 'yellow'),
    ('ReLU16', 'ReLU', 'pink'),
]

# Output layer
dot.node('Output', 'Output\n4', shape = 'box', style = 'filled', color = 'green')

# Add hidden layers
for name, label, color in layers:
    dot.node(name, label, shape = 'box' if 'Dense' in name or 'Dropout' in name else 'ellipse', style = 'filled', color = color)

# Connect nodes
connections = ['Input', 'Dense256', 'ReLU256', 'Dropout256', 'Dense128', 'ReLU128', 'Dropout128',
               'Dense64', 'ReLU64', 'Dropout64', 'Dense32', 'ReLU32', 'Dropout32', 'Dense16', 'ReLU16', 'Output']

for i in range(len(connections)-1):
    dot.edge(connections[i], connections[i+1])

# Render and open the diagram
dot.render('ann_diagram', view = True)