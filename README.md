Topic-Based Text Segmentation using BERT Embeddings
This Rust program reads text from standard input, processes it to identify sentences, and uses BERT sentence embeddings to detect changes in topic or theme. When a topic change is detected based on a similarity threshold, it outputs the accumulated text as a segment. The program is designed to handle large or continuous text inputs efficiently by processing the input in a streaming fashion.
Features

Utilizes the "bert-base-nli-mean-tokens" pre-trained BERT model for generating sentence embeddings.
Detects topic changes by comparing the cosine similarity of sentence embeddings to a running centroid of the current segment.
Processes input asynchronously using tokio, making it suitable for large or continuous text streams.
Outputs segments of text when a topic change is detected, allowing for easy identification of thematic shifts.

Requirements

Rust programming language (version 1.56 or later).
Dependencies:
rust_bert = "0.21.0"
ndarray = "0.15.6"
tokio = { version = "1.38.0", features = ["full"] }
anyhow = "1.0.86"



Installation

Install Rust: If you haven't already, install Rust by following the instructions on rust-lang.org.
Clone the repository: Clone this repository or create a new Rust project and copy the code into your project.
Add dependencies: Ensure your Cargo.toml includes the following dependencies:

[dependencies]
rust_bert = "0.21.0"
ndarray = "0.15.6"
tokio = { version = "1.38.0", features = ["full"] }
anyhow = "1.0.86"


Build the project: Run the following command to build the project:

cargo build

Usage
To run the program, use the following command, replacing input.txt with your text file:
cargo run < input.txt

Alternatively, for continuous input streams (e.g., from another process), you can pipe the input:
cat input.txt | cargo run

The program will process the input and output segments of text when a topic change is detected. Each segment is prefixed with:
----- ЧАСТЬ -----

Notes

The program assumes that sentences end with ., ?, or ! followed by a space. This is a simplification and may not cover all sentence structures.
The program runs in a loop, processing input until the stream ends (for finite inputs) or indefinitely for continuous streams.

Configuration

Threshold: The similarity threshold for determining topic changes is set to 0.5 by default. You can adjust this value in the code to make the segmentation more or less sensitive to topic shifts. A higher threshold will result in more frequent segmentations, while a lower threshold will group more sentences together.

To change the threshold, modify the following line in the code:
let threshold = 0.5;  // Adjust this value as needed

Example
Input
This is the first sentence. It talks about topic A.
Here is another sentence about topic A.
Now, switching to topic B. This sentence is about topic B.

Output
----- ЧАСТЬ -----
This is the first sentence. It talks about topic A.
Here is another sentence about topic A.

----- ЧАСТЬ -----
Now, switching to topic B. This sentence is about topic B.

In this example, the program detects a topic change between the second and third sentences based on the similarity of their embeddings and outputs two separate segments.
Contributing
Contributions are welcome! If you have suggestions for improvements or bug fixes, please submit a pull request or open an issue on the repository.
License
This project is licensed under the MIT License.
