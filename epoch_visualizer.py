import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Batch / Iteration / Epoch", layout="wide")
st.title("Batch Size vs Iteration vs Epoch")
st.caption("Move the slider and watch what happens at every training step")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    n_samples = st.number_input("Number of images (training samples)", 10, 20000, 3200, step=100)
    batch_size = st.number_input("Batch size", 1, int(n_samples), 32, step=1)
    epochs = st.number_input("Number of epochs", 1, 100, 10, step=1)
    shuffle = st.checkbox("Shuffle between epochs", value=True)
    drop_last = st.checkbox("drop_last (skip the final incomplete batch)", value=False)
    framework = st.radio("Tensor layout", ["PyTorch", "TensorFlow"], horizontal=True)

n_samples, batch_size, epochs = int(n_samples), int(batch_size), int(epochs)

# ---------------- Core math ----------------
if drop_last:
    iters_per_epoch = n_samples // batch_size
else:
    iters_per_epoch = math.ceil(n_samples / batch_size)

if iters_per_epoch == 0:
    st.error("Batch size is larger than the dataset and drop_last is on, so there are no batches.")
    st.stop()

total_iters = iters_per_epoch * epochs
last_batch = batch_size if drop_last else n_samples - (iters_per_epoch - 1) * batch_size

c1, c2, c3, c4 = st.columns(4)
c1.metric("Batches per epoch", iters_per_epoch)
c2.metric("Iterations per epoch", iters_per_epoch)
c3.metric("Total iterations", total_iters)
c4.metric("Size of last batch", last_batch)

st.code(
    f"{n_samples} samples / {batch_size} batch size = {n_samples / batch_size:.2f}"
    f"  ->  {iters_per_epoch} iterations per epoch\n"
    f"{iters_per_epoch} iterations x {epochs} epochs = {total_iters} total iterations",
    language="text",
)

# ---------------- Iteration slider ----------------
it = st.slider("Global iteration", 1, total_iters, 1)
epoch_idx = (it - 1) // iters_per_epoch  # 0-based
step_idx = (it - 1) % iters_per_epoch  # 0-based

rng = np.random.default_rng(epoch_idx)
order = rng.permutation(n_samples) if shuffle else np.arange(n_samples)

start = step_idx * batch_size
end = min(start + batch_size, n_samples)
current = order[start:end]
processed = order[:start]

status = np.zeros(n_samples, dtype=int)  # 0 = not yet seen, 1 = processed, 2 = current batch
status[processed] = 1
status[current] = 2

# ---------------- Progress ----------------
p1, p2 = st.columns(2)
with p1:
    st.write(f"**Epoch {epoch_idx + 1} / {epochs}**")
    st.progress((epoch_idx + 1) / epochs)
with p2:
    st.write(f"**Iteration {step_idx + 1} / {iters_per_epoch} (within this epoch)**")
    st.progress((step_idx + 1) / iters_per_epoch)

# ---------------- Grid of samples ----------------
cols = int(math.ceil(math.sqrt(n_samples * 1.6)))
rows = int(math.ceil(n_samples / cols))
z = np.full(rows * cols, np.nan)
z[:n_samples] = status
z = z.reshape(rows, cols)

sample_ids = np.full(rows * cols, np.nan)
sample_ids[:n_samples] = np.arange(n_samples)
sample_ids = sample_ids.reshape(rows, cols)

COLORS = {"unseen": "#d9d9d9", "done": "#4c9be8", "current": "#f26b3a"}
colorscale = [
    [0.0, COLORS["unseen"]], [1 / 3, COLORS["unseen"]],
    [1 / 3, COLORS["done"]], [2 / 3, COLORS["done"]],
    [2 / 3, COLORS["current"]], [1.0, COLORS["current"]],
]

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=z,
        customdata=sample_ids,
        colorscale=colorscale,
        zmin=0,
        zmax=2,
        showscale=False,
        xgap=1,
        ygap=1,
        hovertemplate="Sample #%{customdata:.0f}<extra></extra>",
        showlegend=False,
    )
)
# Dummy traces so the legend explains the colors
for key, label in [
    ("unseen", "Not seen yet in this epoch"),
    ("done", "Already processed in this epoch"),
    ("current", f"Current batch ({len(current)} images)"),
]:
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=COLORS[key], symbol="square"),
            name=label,
        )
    )
fig.update_layout(
    title=f"Each square = one image | Epoch {epoch_idx + 1}, Iteration {step_idx + 1}",
    height=int(min(700, max(300, 900 * rows / cols))),
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5),
)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False, autorange="reversed", scaleanchor="x")
st.plotly_chart(fig, use_container_width=True)

# ---------------- What happens in this step ----------------
st.subheader("What happens in this iteration")
st.markdown(
    f"""
1. Take **{len(current)}** images (batch number {step_idx + 1})
2. **Forward propagation** with the current weights
3. Compute one **mean loss** for the whole batch
4. **Backpropagation** computes the gradients
5. The **optimizer** makes **one weight update** (not {len(current)} updates)
"""
)

if framework == "PyTorch":
    shape = f"({len(current)}, 3, 224, 224)  # (Batch, Channels, Height, Width)"
else:
    shape = f"({len(current)}, 224, 224, 3)  # (Batch, Height, Width, Channels)"
st.code(f"Input tensor shape = {shape}", language="text")

st.caption(
    f"First 10 sample indices in this batch: {current[:10].tolist()}"
    + (" (scattered because of shuffling)" if shuffle else " (consecutive because shuffle is off)")
)

# ---------------- Samples seen so far ----------------
st.subheader("Samples the model has seen so far")
st.write(
    f"The model has processed **{epoch_idx * n_samples + end}** samples in total "
    f"= {epoch_idx} full epoch(s) + {end} images of the current epoch."
)