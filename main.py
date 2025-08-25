import datasets
from langchain_core.documents import Document


def main():

    # Load the dataset
    guest_dataset = datasets.load_dataset(
        "agents-course/unit3-invitees",
        split="train",
    )

    # Convert dataset entries into Document objects
    docs = [
        Document(
            page_content="\n".join([
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}"
            ]),
            metadata={"name": guest["name"]}
        )
        for guest in guest_dataset
    ]

    print(docs[0])


if __name__ == "__main__":
    main()
