import datetime
import os
import uuid

from airflow.decorators import dag, task
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from unstructured.partition.pdf import Element, partition_pdf

client = QdrantClient(url="http://localhost:6333")
MAX_CHUNK_SIZE = 1024  # Maximum number of characters in a chunk
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small", chunk_size=MAX_CHUNK_SIZE
)


@dag(
    dag_id="pdf_etl_pipeline",
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=60),
    description="Convert a PDF into chunks and store them in a vector database",
)
def PdfETLPipeline():
    @task
    def process_pdf_task(output_folder: str, **kwargs) -> str:
        """
        Process a single PDF into chunks and store them in the output folder.
        """
        # Retrieve the pdf_file from the conf parameter
        pdf_file = kwargs["dag_run"].conf.get("pdf_file")
        if not pdf_file:
            raise ValueError("No 'pdf_file' provided in DAG run configuration.")

        output_id = str(uuid.uuid4())
        output_folder = f"{output_folder}/{output_id}"

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(f"{output_folder}/images", exist_ok=True)

        print(f"Processing PDF... {pdf_file} -> {output_folder}, id: {output_id}")

        elements = partition_pdf(
            filename=pdf_file,
            chunking_strategy="by_title",
            max_characters=MAX_CHUNK_SIZE,
            strategy="auto",
            # infer_table_structure=True,
            model_name="yolox",
            # extract_images_in_pdf=True,
            # image_output_dir_path=f"{output_folder}/images",
        )

        for idx, element in enumerate(elements):
            with open(
                f"{output_folder}/{element.metadata.page_number}_{idx}.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(element.text)

        return output_id

    @task
    def process_chunks_task(output_id: str, output_folder: str):
        """
        Process the generated chunks and store them in the database or vector storage.
        """
        print(f"Processing chunks for id ... {output_id}")

        # read chunks from the output folder
        elements = []
        output_folder = f"{output_folder}/{output_id}"
        for file in os.listdir(output_folder):
            if file.endswith(".txt"):
                page_count, idx = file.split("_")
                idx.replace(".txt", "")
                with open(f"{output_folder}/{file}", "r", encoding="utf-8") as f:
                    elements.append({"text": f.read(), "page": page_count, "idx": idx})


        embeddings = embeddings_model.embed_documents(
            [element["text"] for element in elements]
        )

        if not client.collection_exists("pdf_collection"):
            client.create_collection(
                collection_name="pdf_collection",
                vectors_config=VectorParams(
                    size=len(embeddings[0]), distance=Distance.COSINE
                ),
            )

        points = []
        for i, element in enumerate(elements[:]):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={
                    "text": element["text"],
                    "page": element["page"],
                    "idx": element["idx"],
                },
            )
            points.append(point)

        client.upsert(collection_name="pdf_collection", points=points)

    output_folder = "./data/output"

    # Define task dependencies
    process_chunks_task(
        output_id=process_pdf_task(output_folder=output_folder),
        output_folder=output_folder,
    )


dag = PdfETLPipeline()
