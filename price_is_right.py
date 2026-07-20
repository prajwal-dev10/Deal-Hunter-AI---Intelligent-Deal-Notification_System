import logging
import queue
import threading
import time
import gradio as gr
from deal_agent_framework import DealAgentFramework
from log_utils import reformat
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv(override=True)


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def html_for(log_data):
    output = "<br>".join(log_data[-18:])
    return f"""
    <div id="scrollContent" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; background-color: #222229; padding: 10px;">
    {output}
    </div>
    """


def setup_logging(log_queue):
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class App:
    def __init__(self):
        self.agent_framework = None

    def get_agent_framework(self):
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()
        return self.agent_framework

    def run(self):

        theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple"
        )

        with gr.Blocks(
           title="DealHunter AI | Autonomous Deal Intelligence",
           theme=theme,
           fill_width=True
       ) as ui:
 
           log_data = gr.State([])
 
 
           def table_for(opps):
               return [
                   [
                       opp.deal.product_description,
                       f"${opp.deal.price:.2f}",
                       f"${opp.estimate:.2f}",
                       f"${opp.discount:.2f}",
                       opp.deal.url,
                   ]
                   for opp in opps
               ]
 
 
           def update_output(log_data, log_queue, result_queue):
 
               initial_result = table_for(
                   self.get_agent_framework().memory
               )
 
               final_result = None
 
               while True:
 
                   try:
                       message = log_queue.get_nowait()
 
                       log_data.append(
                           reformat(message)
                       )
 
                       yield (
                           log_data,
                           html_for(log_data),
                           final_result or initial_result
                       )
 
 
                   except queue.Empty:
 
                       try:
 
                           final_result = result_queue.get_nowait()
 
                           yield (
                               log_data,
                               html_for(log_data),
                               final_result
                           )
 
                       except queue.Empty:
 
                           if final_result is not None:
                               break
 
                           time.sleep(0.1)
 
 
 
           def get_plot():
 
               documents, vectors, colors = (
                   DealAgentFramework.get_plot_data(
                       max_datapoints=800
                   )
               )
 
 
               fig = go.Figure(
 
                   data=[
 
                       go.Scatter3d(
 
                           x=vectors[:,0],
                           y=vectors[:,1],
                           z=vectors[:,2],
 
                           mode="markers",
 
                           marker=dict(
                               size=3,
                               color=colors,
                               opacity=0.8
                           )
                       )
                   ]
               )
 
 
               fig.update_layout(
 
                   title="RAG Knowledge Vector Space",
 
                   scene=dict(
 
                       xaxis_title="Embedding X",
                       yaxis_title="Embedding Y",
                       zaxis_title="Embedding Z",
 
                       aspectratio=dict(
                           x=2,
                           y=2,
                           z=1
                       ),
 
                       camera=dict(
                           eye=dict(
                               x=1.6,
                               y=1.6,
                               z=0.8
                           )
                       )
                   ),
 
 
                   height=450,
 
                   margin=dict(
                       l=5,
                       r=5,
                       b=5,
                       t=40
                   )
               )
 
 
               return fig
 
 
 
           def do_run():
 
               new_opportunities = (
                   self.get_agent_framework().run()
               )
 
               return table_for(
                   new_opportunities
               )
 
 
 
           def run_with_logging(initial_log_data):
 
               log_queue = queue.Queue()
               result_queue = queue.Queue()
 
 
               setup_logging(log_queue)
 
 
               def worker():
 
                   result = do_run()
 
                   result_queue.put(
                       result
                   )
 
 
               thread = threading.Thread(
                   target=worker
               )
 
               thread.start()
 
 
               for data, logs, table in update_output(
                   initial_log_data,
                   log_queue,
                   result_queue
               ):
 
                   yield data, logs, table,get_plot()
 
 
 
           def do_select(selected_index):
 
               opportunities = (
                   self.get_agent_framework().memory
               )
 
               row = selected_index.index[0]
 
               opportunity = opportunities[row]
 
 
               self.get_agent_framework().planner.messenger.alert(
                   opportunity
               )
 
 
 
           # =========================
           # HERO HEADER
           # =========================
 
 
           gr.Markdown(
           """
 
           <div style="
               text-align:center;
               padding:25px;
           ">
 
 
           <h1 style="
               font-size:48px;
               margin-bottom:5px;
           ">
 
           🛒 DealHunter AI
 
           </h1>
 
 
           <h3 style="
               font-weight:400;
               color:#666;
           ">
 
           Autonomous Multi-Agent Deal Intelligence System
 
           </h3>
 
 
           <p style="
               font-size:17px;
           ">
 
           An AI-powered agent framework that discovers online deals,
           evaluates product value, reasons with LLMs,
           and identifies high-value opportunities using
           Retrieval Augmented Generation.
 
           </p>
 
 
           </div>
 
           """
           )
 
 
           gr.Markdown(
           """
 
           <div style="
           text-align:center;
           padding:10px;
           font-size:16px;
           ">
 
 
           🤖 LLM Agents
 
           &nbsp;&nbsp;|&nbsp;&nbsp;
 
           🔎 Autonomous Web Discovery
 
           &nbsp;&nbsp;|&nbsp;&nbsp;
 
           🧠 RAG Pipeline
 
           &nbsp;&nbsp;|&nbsp;&nbsp;
 
           📚 Chroma Vector Database
 
           &nbsp;&nbsp;|&nbsp;&nbsp;
 
           ⚡ AI Decision Engine
 
 
           </div>
 
           """
           )
 
 
 
           # =========================
           # DEAL RESULTS
           # =========================
 
 
           gr.Markdown(
           "## 🔥 Discovered Opportunities"
           )
 
 
           opportunities_dataframe = gr.Dataframe(
 
               headers=[
 
                   "🛍 Product Description",
 
                   "💵 Deal Price",
 
                   "📈 Estimated Value",
 
                   "🔥 Discount",
 
                   "🔗 Source"
 
               ],
 
               wrap=True,
 
               column_widths=[
                   8,
                   2,
                   2,
                   2,
                   4
               ],
 
               row_count=10,
 
               max_height=450,
 
               interactive=False
           )
 
 
 
           # =========================
           # MONITORING
           # =========================
 
 
           with gr.Row():
 
 
               with gr.Column():
 
                   gr.Markdown(
                       "## 📝 Agent Execution Trace"
                   )
 
                   logs = gr.HTML()
 
 
 
               with gr.Column():
 
                   gr.Markdown(
                       "## 🧬 RAG Embedding Visualization"
                   )
 
 
                   plot = gr.Plot(
                       value=get_plot(),
                       show_label=False
                   )
 
 
 
           # =========================
           # AUTOMATION
           # =========================
 
 
           ui.load(
 
               run_with_logging,
 
               inputs=[
                   log_data
               ],
 
               outputs=[
                   log_data,
                   logs,
                   opportunities_dataframe,
                   plot
               ]
 
           )
 
 
 
           timer = gr.Timer(
               value=60,
               active=True
           )
 
 
           timer.tick(
 
               run_with_logging,
 
               inputs=[
                   log_data
               ],
 
               outputs=[
                   log_data,
                   logs,
                   opportunities_dataframe,
                   plot
               ]
 
           )
 
 
           opportunities_dataframe.select(
               do_select
           )
 
 
 
        ui.launch(
            share=False,
            inbrowser=True
        )


if __name__ == "__main__":
    App().run()
