import os
import datetime
import pendulum

from airflow.decorators import dag, task


@dag(
    dag_id="pdf_etl_pipeline",
    schedule_interval=None,
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=60),
    description="Convert a PDF into chunks and store them in a vector database",
)
def PdfETLPipeline():
    @task
    def process_pdf_task(pdf_file: str, output_folder: str):
        """
        Process a single PDF into chunks and store them in the output folder.
        """
        print(f"Processing PDF... {pdf_file} -> {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
        # Add your PDF processing logic here.

    @task
    def process_chunks_task():
        """
        Process the generated chunks and store them in the database or vector storage.
        """
        print("Processing chunks...")
        # Add your chunk processing and storage logic here.

    # Define parameters
    pdf_file = "/path/to/project/data/pdfs/sample.pdf"
    output_folder = "./data/output"

    # Define task dependencies
    process_pdf_task(pdf_file=pdf_file, output_folder=output_folder) >> process_chunks_task()


dag = PdfETLPipeline()
